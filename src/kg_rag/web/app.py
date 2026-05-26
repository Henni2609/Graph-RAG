from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import asyncio

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kg_rag.config import RagConfig
from kg_rag.llm import stream_chat_tokens
from kg_rag.components.reranker import _get_cross_encoder
from kg_rag.logging import logger
from kg_rag.neo4j_store import Neo4jGraphStore, stable_id
from kg_rag.pipelines.indexing import IndexingPipeline, _get_doc_embedder, _resolve_device
from kg_rag.pipelines.query import (
    ANSWER_SYSTEM_PROMPT,
    QueryPipeline,
    RetrievalResult,
    _GENERATION_ERROR,
    embed_query,
    invalidate_embedding_meta,
    sanitize_citations,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = Path(tempfile.gettempdir()) / "kg-rag-uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{16,64}$")
JOB_TTL_SECONDS = 3600
JOB_STATUS = Literal["queued", "running", "done", "error"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    hops: int | None = Field(default=None, ge=1, le=3)


class SessionEndRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)


@dataclass
class FileEntry:
    filename: str
    document_id: str
    status: Literal["queued", "parsing", "indexing", "done", "error"] = "queued"
    error: str | None = None


@dataclass
class JobState:
    id: str
    session_id: str
    files: list[FileEntry]
    status: JOB_STATUS
    step: str = "queued"
    current: int = 0
    total: int = 0
    chunks_indexed: int = 0
    error: str | None = None
    estimated_seconds: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    graph: dict[str, Any] | None = None
    future: Future | None = None
    tmp_dir: Path | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "files": [
                {"filename": f.filename, "document_id": f.document_id, "status": f.status, "error": f.error}
                for f in self.files
            ],
            "status": self.status,
            "step": self.step,
            "current": self.current,
            "total": self.total,
            "chunks_indexed": self.chunks_indexed,
            "error": self.error,
            "estimated_seconds": self.estimated_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "graph": self.graph,
        }


INDEXING_JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
MANIFEST_LOCK = threading.Lock()
JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("JOB_CONCURRENCY", "1")),
    thread_name_prefix="indexing-job",
)

_PDF_PAGES_CACHE: dict[str, list[dict[str, Any]]] = {}
_PDF_PAGES_CACHE_LOCK = threading.Lock()


def _require_session_id(header_value: str | None) -> str:
    if not header_value or not SESSION_ID_PATTERN.match(header_value):
        raise HTTPException(status_code=400, detail="Ungültige oder fehlende Session-ID")
    return header_value


def _estimate_seconds(byte_count: int) -> int:
    # Rough heuristic: 30s base + 4s per MB. Calibrated against the
    # observed 2–3min runtime for a ~5 MB / 45-page PDF.
    return int(30 + (byte_count / 1_000_000) * 4)


def _evict_old_jobs(now: float) -> None:
    expired = [
        job_id
        for job_id, state in INDEXING_JOBS.items()
        if state.finished_at is not None and (now - state.finished_at) > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        INDEXING_JOBS.pop(job_id, None)


def _update_job(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        state = INDEXING_JOBS.get(job_id)
        if state is None:
            return
        for key, value in fields.items():
            setattr(state, key, value)


def create_app(config: RagConfig | None = None) -> FastAPI:
    app_config = config or RagConfig.from_env()
    app = FastAPI(title="Graph RAG", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def _warm_embedder() -> None:
        threading.Thread(
            target=_warmup_embedder,
            args=(app_config.embedding_model, app_config.embedding_batch_size, app_config.embedding_device),
            name="embedder-warmup",
            daemon=True,
        ).start()
        threading.Thread(target=_warmup_pdf_libs, name="pdf-libs-warmup", daemon=True).start()
        if app_config.reranker_enabled:
            threading.Thread(
                target=_warmup_reranker,
                args=(app_config.reranker_model, _resolve_device(app_config.embedding_device)),
                name="reranker-warmup",
                daemon=True,
            ).start()
        threading.Thread(
            target=_warmup_llm_extraction,
            args=(app_config,),
            name="llm-extraction-warmup",
            daemon=True,
        ).start()

    @app.on_event("startup")
    async def _warm_llm_stream() -> None:
        try:
            gk: dict[str, Any] = {
                "max_tokens": 1,
                "temperature": 0,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            async for _ in stream_chat_tokens(app_config.llm, "ping", "1", generation_kwargs=gk):
                break
            logger.info("LLM stream connection warmup complete")
        except Exception:
            logger.exception("LLM stream connection warmup failed")

    @app.on_event("shutdown")
    def _shutdown_jobs() -> None:
        JOB_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/graph")
    def graph(x_session_id: str | None = Header(default=None)) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        return _fetch_graph(app_config, session_id)

    @app.post("/api/upload", status_code=202)
    async def upload(
        files: list[UploadFile] = File(...),
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        return await _enqueue_upload(files, app_config, session_id)

    @app.get("/api/document/{document_id}/text")
    def document_text(
        document_id: str,
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        if not DOCUMENT_ID_PATTERN.match(document_id):
            raise HTTPException(status_code=400, detail="Ungültige Dokument-ID")
        path = _resolve_pdf_file(session_id, document_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
        with _PDF_PAGES_CACHE_LOCK:
            cached = _PDF_PAGES_CACHE.get(document_id)
        if cached is None:
            try:
                cached = _extract_pdf_pages(path, ocr_language=app_config.ocr_language)
            except Exception as exc:
                logger.exception("PDF-Textextraktion fehlgeschlagen für %s", path)
                raise HTTPException(status_code=500, detail=f"Text konnte nicht extrahiert werden: {exc}") from exc
            with _PDF_PAGES_CACHE_LOCK:
                _PDF_PAGES_CACHE[document_id] = cached
        return {"document_id": document_id, "title": path.name, "pages": cached}

    @app.delete("/api/document/{document_id}")
    def delete_document(
        document_id: str,
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        if not DOCUMENT_ID_PATTERN.match(document_id):
            raise HTTPException(status_code=400, detail="Ungültige Dokument-ID")
        store = Neo4jGraphStore(app_config.neo4j)
        try:
            store.delete_document(session_id, document_id)
        except Exception as exc:
            logger.exception("Dokument-Löschung fehlgeschlagen für %s", document_id)
            raise HTTPException(status_code=500, detail=f"Löschen fehlgeschlagen: {exc}") from exc
        finally:
            store.close()
        _remove_pdf_manifest(session_id, document_id)
        invalidate_embedding_meta(session_id)
        with _PDF_PAGES_CACHE_LOCK:
            _PDF_PAGES_CACHE.pop(document_id, None)
        return {"deleted": document_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(
        job_id: str,
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        with JOBS_LOCK:
            state = INDEXING_JOBS.get(job_id)
            if state is None or state.session_id != session_id:
                raise HTTPException(status_code=404, detail="Job nicht gefunden")
            return state.snapshot()

    @app.post("/api/query")
    def query(
        request: QueryRequest,
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        return _handle_query(request, app_config, session_id)

    @app.post("/api/query/stream")
    async def query_stream(
        request: QueryRequest,
        x_session_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        session_id = _require_session_id(x_session_id)
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Frage darf nicht leer sein")

        async def generate() -> Any:
            t0 = time.perf_counter()
            pipeline = QueryPipeline(app_config)
            try:
                retrieval: RetrievalResult = await asyncio.to_thread(
                    pipeline.retrieve,
                    question,
                    top_k=request.top_k,
                    hops=request.hops,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.exception("Stream query retrieval failed")
                yield f"event: error\ndata: {json.dumps({'detail': f'Anfrage fehlgeschlagen: {exc}'})}\n\n"
                return
            finally:
                try:
                    pipeline.store.close()
                except Exception:
                    pass

            logger.info(f"TIMING stream_retrieval: {time.perf_counter() - t0:.3f}s")
            meta_data = {
                "query_entities": retrieval.query_entities,
                "vector_chunks": len(retrieval.vector_documents),
                "graph_chunks": len(retrieval.graph_documents),
                "citations": retrieval.citations,
            }
            yield f"event: meta\ndata: {json.dumps(meta_data)}\n\n"

            if retrieval.early_answer is not None:
                yield f"event: final\ndata: {json.dumps({'answer': retrieval.early_answer})}\n\n"
                logger.info(f"TIMING stream_total: {time.perf_counter() - t0:.3f}s (early exit)")
                return

            gk: dict[str, Any] = {
                "temperature": 0.2,
                "max_tokens": app_config.answer_max_tokens,
                "timeout": app_config.answer_timeout_seconds,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            prompt = f"Kontext:\n{retrieval.context}\n\nFrage:\n{question}"
            tokens: list[str] = []
            first_token = True
            try:
                async for delta in stream_chat_tokens(app_config.llm, ANSWER_SYSTEM_PROMPT, prompt, generation_kwargs=gk):
                    if first_token:
                        logger.info(f"TIMING stream_ttft: {time.perf_counter() - t0:.3f}s")
                        first_token = False
                    tokens.append(delta)
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"
            except Exception as exc:
                logger.warning(f"Stream answer generation failed: {exc}", exc_info=True)
                yield f"event: final\ndata: {json.dumps({'answer': _GENERATION_ERROR})}\n\n"
                return

            valid_indexes = {c["index"] for c in retrieval.citations}
            full_answer = sanitize_citations("".join(tokens), valid_indexes)
            yield f"event: final\ndata: {json.dumps({'answer': full_answer})}\n\n"
            logger.info(f"TIMING stream_total: {time.perf_counter() - t0:.3f}s")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/session/end")
    def session_end(payload: SessionEndRequest) -> dict[str, Any]:
        session_id = _require_session_id(payload.session_id)
        store = Neo4jGraphStore(app_config.neo4j)
        try:
            store.delete_session(session_id)
        except Exception as exc:
            logger.exception("Session cleanup failed")
            raise HTTPException(status_code=500, detail=f"Session-Cleanup fehlgeschlagen: {exc}") from exc
        finally:
            store.close()
        invalidate_embedding_meta(session_id)
        _cleanup_session_uploads(session_id)
        return {"deleted": session_id}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _enqueue_upload(upload_files: list[UploadFile], config: RagConfig, session_id: str) -> dict[str, Any]:
    # Validate and read all files first — reject the whole batch if any file fails.
    validated: list[tuple[str, bytes]] = []
    seen_names: set[str] = set()
    for uf in upload_files:
        filename = uf.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Nur PDF-Dateien werden akzeptiert: {filename}")
        contents = await uf.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"Datei ist leer: {filename}")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Datei zu groß (max. 50 MB): {filename}")
        safe_name = Path(filename).name or "upload.pdf"
        if safe_name in seen_names:
            raise HTTPException(status_code=400, detail=f"Doppelter Dateiname im Upload: {safe_name}")
        seen_names.add(safe_name)
        validated.append((safe_name, contents))

    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_entries: list[FileEntry] = []
    pdf_paths: list[Path] = []
    manifest_entries: list[tuple[str, str]] = []
    estimated_total = 0

    for safe_name, contents in validated:
        pdf_path = session_dir / safe_name
        pdf_path.write_bytes(contents)
        document_id = stable_id(f"{session_id}|{str(pdf_path.resolve())}")
        file_entries.append(FileEntry(filename=safe_name, document_id=document_id, status="queued"))
        pdf_paths.append(pdf_path)
        manifest_entries.append((document_id, safe_name))
        estimated_total += _estimate_seconds(len(contents))

    _record_pdf_manifest(session_id, manifest_entries)

    job_id = uuid.uuid4().hex
    state = JobState(
        id=job_id,
        session_id=session_id,
        files=file_entries,
        status="queued",
        step="queued",
        estimated_seconds=estimated_total,
        tmp_dir=session_dir,
    )
    with JOBS_LOCK:
        _evict_old_jobs(time.time())
        INDEXING_JOBS[job_id] = state

    future = JOB_EXECUTOR.submit(_run_indexing_job, job_id, pdf_paths, config)
    state.future = future

    return {
        "job_id": job_id,
        "files": [{"filename": e.filename, "document_id": e.document_id} for e in file_entries],
        "status": "queued",
        "estimated_seconds": estimated_total,
    }


def _run_indexing_job(job_id: str, pdf_paths: list[Path], config: RagConfig) -> None:
    with JOBS_LOCK:
        state = INDEXING_JOBS.get(job_id)
        if state is None:
            return
        session_id = state.session_id

    _update_job(job_id, status="running", step="parsing", current=0, total=0)

    def progress(step: str, current: int, total: int) -> None:
        _update_job(job_id, step=step, current=current, total=total)

    def on_file(document_id: str, status: str) -> None:
        with JOBS_LOCK:
            s = INDEXING_JOBS.get(job_id)
            if s is None:
                return
            for entry in s.files:
                if entry.document_id == document_id:
                    entry.status = status
                    break

    pipeline = IndexingPipeline(config)
    try:
        ocr_map: dict[str, dict[int, str]] = {}

        def on_pages_loaded(docs: list) -> None:
            from kg_rag.compat import document_content, document_meta
            for doc in docs:
                meta = document_meta(doc)
                if meta.get("extraction") != "ocr":
                    continue
                doc_id = str(meta.get("document_id", ""))
                page = int(meta.get("page_number", 0))
                if doc_id and page:
                    ocr_map.setdefault(doc_id, {})[page] = document_content(doc)

        chunk_count = pipeline.run(
            pdf_paths,
            session_id=session_id,
            progress=progress,
            on_file=on_file,
            on_pages_loaded=on_pages_loaded,
        )
        _purge_orphan_chunks(config, session_id)
        invalidate_embedding_meta(session_id)
        graph_data = _fetch_graph_unsafe(config, session_id)
        with JOBS_LOCK:
            doc_id_to_path = {
                e.document_id: p
                for e, p in zip(INDEXING_JOBS[job_id].files, pdf_paths)
            } if job_id in INDEXING_JOBS else {}
        for doc_id, pdf_path in doc_id_to_path.items():
            threading.Thread(
                target=_precache_pdf_pages,
                args=(doc_id, pdf_path, config.ocr_language, ocr_map.get(doc_id, {})),
                name=f"pdf-precache-{doc_id[:8]}",
                daemon=True,
            ).start()
        _update_job(
            job_id,
            status="done",
            step="done",
            chunks_indexed=chunk_count,
            current=chunk_count,
            total=chunk_count,
            graph=graph_data,
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("Indexing job %s failed", job_id)
        with JOBS_LOCK:
            s = INDEXING_JOBS.get(job_id)
            if s:
                for entry in s.files:
                    if entry.status != "done":
                        entry.status = "error"
                        entry.error = str(exc)[:200]
        _update_job(
            job_id,
            status="error",
            step="error",
            error=str(exc)[:500],
            finished_at=time.time(),
        )
    finally:
        try:
            pipeline.store.close()
        except Exception:
            logger.exception("Failed to close store for job %s", job_id)


def _manifest_path(session_id: str) -> Path:
    return UPLOADS_DIR / session_id / "index.json"


def _record_pdf_manifest(session_id: str, entries: list[tuple[str, str]]) -> None:
    path = _manifest_path(session_id)
    with MANIFEST_LOCK:
        try:
            manifest: dict[str, str] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        except Exception:
            manifest = {}
        for document_id, filename in entries:
            manifest[document_id] = filename
        try:
            path.write_text(json.dumps(manifest), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write PDF manifest for session %s", session_id)


def _resolve_pdf_file(session_id: str, document_id: str) -> Path | None:
    session_dir = UPLOADS_DIR / session_id
    if not session_dir.is_dir():
        return None
    try:
        manifest: dict[str, str] = json.loads(_manifest_path(session_id).read_text("utf-8"))
    except Exception:
        return None
    filename = manifest.get(document_id)
    if not filename:
        return None
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        return None
    candidate = (session_dir / filename).resolve()
    try:
        candidate.relative_to(session_dir.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _remove_pdf_manifest(session_id: str, document_id: str) -> None:
    path = _manifest_path(session_id)
    pdf_filename: str | None = None
    with MANIFEST_LOCK:
        try:
            manifest: dict[str, str] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        except Exception:
            manifest = {}
        pdf_filename = manifest.pop(document_id, None)
        try:
            path.write_text(json.dumps(manifest), encoding="utf-8")
        except Exception:
            logger.exception("Failed to update PDF manifest for session %s", session_id)
    # Delete the file outside the lock to avoid holding it during I/O.
    if pdf_filename:
        session_dir = UPLOADS_DIR / session_id
        if "/" not in pdf_filename and "\\" not in pdf_filename and not pdf_filename.startswith(".."):
            candidate = (session_dir / pdf_filename).resolve()
            try:
                candidate.relative_to(session_dir.resolve())
                candidate.unlink(missing_ok=True)
            except ValueError:
                pass
            except Exception:
                logger.exception("Failed to delete PDF file for document %s", document_id)


def _warmup_pdf_libs() -> None:
    try:
        from haystack.components.converters.pypdf import PyPDFToDocument  # noqa: F401
    except Exception:
        pass
    try:
        from pypdf import PdfReader  # noqa: F401
    except Exception:
        pass
    try:
        import pytesseract  # noqa: F401
    except Exception:
        pass


def _precache_pdf_pages(document_id: str, path: Path, ocr_language: str, precomputed_ocr: dict[int, str] | None = None) -> None:
    try:
        pages = _extract_pdf_pages(path, ocr_language=ocr_language, precomputed_ocr=precomputed_ocr)
        with _PDF_PAGES_CACHE_LOCK:
            _PDF_PAGES_CACHE[document_id] = pages
        logger.info("PDF-Seiten vorgeladen für Dokument %s (%d Seiten)", document_id[:8], len(pages))
    except Exception:
        logger.exception("PDF-Vorladen fehlgeschlagen für Dokument %s", document_id[:8])


def _extract_pdf_pages(
    path: Path,
    ocr_language: str = "deu+eng",
    precomputed_ocr: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    # --- pypdf text extraction (always run, cheap) ---
    text_by_page: dict[int, str] = {}
    try:
        from haystack.components.converters.pypdf import PyPDFToDocument
        from kg_rag.compat import document_content
        from kg_rag.pipelines.indexing import _clean_page_text

        converter = PyPDFToDocument()
        result = converter.run(sources=[path])
        if result["documents"]:
            full_text = document_content(result["documents"][0])
            for i, chunk in enumerate(full_text.split("\x0c")):
                txt = _clean_page_text(chunk).strip()
                if txt:
                    text_by_page[i + 1] = txt
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                text_by_page[idx] = text

    # --- OCR: reuse results from ingestion if available, else run ---
    if precomputed_ocr is not None:
        ocr_by_page: dict[int, str] = precomputed_ocr
    else:
        ocr_by_page = {}
        try:
            import pytesseract
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for page_idx, page in enumerate(reader.pages, start=1):
                page_texts: list[str] = []
                try:
                    images = page.images
                except Exception:
                    continue
                for img_file in images:
                    try:
                        text = pytesseract.image_to_string(img_file.image, lang=ocr_language)
                        if text.strip():
                            page_texts.append(text.strip())
                    except pytesseract.TesseractNotFoundError:
                        break
                    except Exception:
                        continue
                if page_texts:
                    ocr_by_page[page_idx] = "\n".join(page_texts)
        except ImportError:
            pass

    # --- merge: combine pypdf text + OCR per page ---
    pages: list[dict[str, Any]] = []
    for page_idx in sorted(set(text_by_page) | set(ocr_by_page)):
        parts = []
        if page_idx in text_by_page:
            parts.append(text_by_page[page_idx])
        if page_idx in ocr_by_page:
            parts.append(ocr_by_page[page_idx])
        pages.append({"page_number": page_idx, "text": "\n\n".join(parts)})
    return pages


def _cleanup_session_uploads(session_id: str) -> None:
    session_dir = UPLOADS_DIR / session_id
    if not session_dir.exists():
        return
    try:
        shutil.rmtree(session_dir)
    except Exception:
        logger.exception("Failed to clean uploads for session %s", session_id)


def _purge_orphan_chunks(config: RagConfig, session_id: str) -> None:
    manifest_path = _manifest_path(session_id)
    if not manifest_path.exists():
        return
    try:
        manifest: dict[str, str] = json.loads(manifest_path.read_text("utf-8"))
    except Exception:
        return
    if not manifest:
        return
    store = Neo4jGraphStore(config.neo4j)
    try:
        store.delete_orphan_chunks(session_id, set(manifest.keys()))
    except Exception:
        logger.exception("Orphan-chunk cleanup failed for session %s", session_id)
    finally:
        store.close()


def _handle_query(request: QueryRequest, config: RagConfig, session_id: str) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Frage darf nicht leer sein")

    pipeline = QueryPipeline(config)
    try:
        result = pipeline.run(
            question,
            top_k=request.top_k,
            hops=request.hops,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Anfrage fehlgeschlagen: {exc}") from exc
    finally:
        pipeline.store.close()

    return {
        "answer": result.answer,
        "query_entities": result.query_entities,
        "vector_chunks": len(result.vector_documents),
        "graph_chunks": len(result.graph_documents),
        "context": result.context,
        "citations": result.citations,
    }


def _fetch_graph(config: RagConfig, session_id: str) -> dict[str, Any]:
    try:
        return _fetch_graph_unsafe(config, session_id)
    except Exception as exc:
        logger.exception("Graph fetch failed")
        raise HTTPException(status_code=500, detail=f"Graph konnte nicht geladen werden: {exc}") from exc


def _fetch_graph_unsafe(config: RagConfig, session_id: str) -> dict[str, Any]:
    store = Neo4jGraphStore(config.neo4j)
    try:
        return store.fetch_entity_graph(session_id=session_id)
    finally:
        store.close()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    config = RagConfig.from_env()
    uvicorn.run(create_app(config), host=host, port=port)


def _reset_uploads() -> None:
    if not UPLOADS_DIR.exists():
        return
    try:
        shutil.rmtree(UPLOADS_DIR)
    except Exception:
        logger.exception("Failed to clean uploads dir on startup")


def _warmup_embedder(model: str, batch_size: int, device: str) -> None:
    try:
        embed_query("warmup", model=model, device=device)
        _get_doc_embedder(model, batch_size, _resolve_device(device))
        logger.info(f"Embedder warmup complete for {model} on {_resolve_device(device)}")
    except Exception:
        logger.exception("Embedder warmup failed")


def _warmup_reranker(model: str, device: str) -> None:
    try:
        encoder = _get_cross_encoder(model, device)
        encoder.predict([("warm", "up")], show_progress_bar=False)
        logger.info(f"Reranker warmup complete for {model} on {device}")
    except Exception:
        logger.exception("Reranker warmup failed")


def _warmup_llm_extraction(config: RagConfig) -> None:
    try:
        from kg_rag.llm import create_chat_generator, run_chat
        gen = create_chat_generator(
            config.llm,
            model=config.llm.extraction_model,
            timeout=15,
            max_retries=1,
        )
        run_chat(
            gen,
            "ping",
            "1",
            generation_kwargs={"max_tokens": 1, "temperature": 0, "extra_body": {"thinking": {"type": "disabled"}}},
        )
        logger.info("LLM extraction connection warmup complete")
    except Exception:
        logger.exception("LLM extraction connection warmup failed")


def _reset_graph(config: RagConfig) -> None:
    store = Neo4jGraphStore(config.neo4j)
    try:
        store.clear()
        logger.info("Graph database cleared for new server run")
    except Exception:
        logger.exception("Failed to clear graph database on startup")
    finally:
        store.close()

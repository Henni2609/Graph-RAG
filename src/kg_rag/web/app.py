from __future__ import annotations

import json
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

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kg_rag.config import RagConfig
from kg_rag.logging import logger
from kg_rag.neo4j_store import Neo4jGraphStore, stable_id
from kg_rag.pipelines.indexing import IndexingPipeline
from kg_rag.pipelines.query import QueryPipeline


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
class JobState:
    id: str
    session_id: str
    filename: str
    status: JOB_STATUS
    document_id: str = ""
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "filename": self.filename,
            "document_id": self.document_id,
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
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexing-job")


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
            args=(app_config.embedding_model,),
            name="embedder-warmup",
            daemon=True,
        ).start()

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
        file: UploadFile = File(...),
        x_session_id: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session_id = _require_session_id(x_session_id)
        return await _enqueue_upload(file, app_config, session_id)

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
        try:
            pages = _extract_pdf_pages(path)
        except Exception as exc:
            logger.exception("PDF-Textextraktion fehlgeschlagen für %s", path)
            raise HTTPException(status_code=500, detail=f"Text konnte nicht extrahiert werden: {exc}") from exc
        return {"document_id": document_id, "title": path.name, "pages": pages}

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
        _cleanup_session_uploads(session_id)
        return {"deleted": session_id}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _enqueue_upload(file: UploadFile, config: RagConfig, session_id: str) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien werden akzeptiert")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Datei ist leer")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Datei zu groß (max. 50 MB)")

    safe_name = Path(filename).name or "upload.pdf"
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = session_dir / safe_name
    pdf_path.write_bytes(contents)
    resolved_source = str(pdf_path.resolve())
    document_id = stable_id(f"{session_id}|{resolved_source}")
    _record_pdf_manifest(session_id, document_id, safe_name)

    job_id = uuid.uuid4().hex
    estimated = _estimate_seconds(len(contents))

    state = JobState(
        id=job_id,
        session_id=session_id,
        filename=safe_name,
        document_id=document_id,
        status="queued",
        step="queued",
        estimated_seconds=estimated,
    )
    with JOBS_LOCK:
        _evict_old_jobs(time.time())
        INDEXING_JOBS[job_id] = state

    future = JOB_EXECUTOR.submit(_run_indexing_job, job_id, pdf_path, config)
    state.future = future

    return {
        "job_id": job_id,
        "filename": safe_name,
        "document_id": document_id,
        "status": "queued",
        "estimated_seconds": estimated,
    }


def _run_indexing_job(job_id: str, pdf_path: Path, config: RagConfig) -> None:
    with JOBS_LOCK:
        state = INDEXING_JOBS.get(job_id)
        if state is None:
            return
        session_id = state.session_id

    _update_job(job_id, status="running", step="parsing", current=0, total=0)

    def progress(step: str, current: int, total: int) -> None:
        _update_job(job_id, step=step, current=current, total=total)

    pipeline = IndexingPipeline(config)
    try:
        chunk_count = pipeline.run(
            [pdf_path],
            session_id=session_id,
            progress=progress,
        )
        graph_data = _fetch_graph_unsafe(config, session_id)
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


def _record_pdf_manifest(session_id: str, document_id: str, filename: str) -> None:
    path = _manifest_path(session_id)
    try:
        manifest: dict[str, str] = json.loads(path.read_text("utf-8")) if path.exists() else {}
    except Exception:
        manifest = {}
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


def _extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": idx, "text": text})
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

    _purge_orphan_chunks(config, session_id)
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


def _warmup_embedder(model: str) -> None:
    try:
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder

        embedder = SentenceTransformersDocumentEmbedder(model=model)
        embedder.warm_up()
        logger.info("Embedder warmup complete for %s", model)
    except Exception:
        logger.exception("Embedder warmup failed")


def _reset_graph(config: RagConfig) -> None:
    store = Neo4jGraphStore(config.neo4j)
    try:
        store.clear()
        logger.info("Graph database cleared for new server run")
    except Exception:
        logger.exception("Failed to clear graph database on startup")
    finally:
        store.close()

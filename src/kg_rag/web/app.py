from __future__ import annotations

import re
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
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines.indexing import IndexingPipeline
from kg_rag.pipelines.query import QueryPipeline


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
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
    step: str = "queued"
    current: int = 0
    total: int = 0
    chunks_indexed: int = 0
    error: str | None = None
    estimated_seconds: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    graph: dict[str, Any] | None = None
    tmp_dir: Path | None = None
    future: Future | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "filename": self.filename,
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
    tmp_dir = Path(tempfile.mkdtemp(prefix="kg-rag-upload-"))
    tmp_path = tmp_dir / safe_name
    tmp_path.write_bytes(contents)

    job_id = uuid.uuid4().hex
    estimated = _estimate_seconds(len(contents))

    state = JobState(
        id=job_id,
        session_id=session_id,
        filename=safe_name,
        status="queued",
        step="queued",
        estimated_seconds=estimated,
        tmp_dir=tmp_dir,
    )
    with JOBS_LOCK:
        _evict_old_jobs(time.time())
        INDEXING_JOBS[job_id] = state

    future = JOB_EXECUTOR.submit(_run_indexing_job, job_id, tmp_path, config)
    state.future = future

    return {
        "job_id": job_id,
        "filename": safe_name,
        "status": "queued",
        "estimated_seconds": estimated,
    }


def _run_indexing_job(job_id: str, tmp_path: Path, config: RagConfig) -> None:
    with JOBS_LOCK:
        state = INDEXING_JOBS.get(job_id)
        if state is None:
            return
        session_id = state.session_id
        tmp_dir = state.tmp_dir

    _update_job(job_id, status="running", step="parsing", current=0, total=0)

    def progress(step: str, current: int, total: int) -> None:
        _update_job(job_id, step=step, current=current, total=total)

    pipeline = IndexingPipeline(config)
    try:
        chunk_count = pipeline.run(
            [tmp_path],
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
        if tmp_dir is not None and tmp_dir.exists():
            try:
                for child in tmp_dir.iterdir():
                    child.unlink()
                tmp_dir.rmdir()
            except Exception:
                logger.exception("Failed to clean tmp_dir for job %s", job_id)


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
    _reset_graph(config)
    uvicorn.run(create_app(config), host=host, port=port)


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

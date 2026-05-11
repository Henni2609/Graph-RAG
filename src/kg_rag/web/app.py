from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kg_rag.config import RagConfig
from kg_rag.logging import logger
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines.indexing import IndexingPipeline


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(config: RagConfig | None = None) -> FastAPI:
    app_config = config or RagConfig.from_env()
    app = FastAPI(title="Graph RAG", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/graph")
    def graph() -> dict[str, Any]:
        return _fetch_graph(app_config)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        return await _handle_upload(file, app_config)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _handle_upload(file: UploadFile, config: RagConfig) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien werden akzeptiert")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Datei ist leer")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Datei zu groß (max. 50 MB)")

    chunk_count = _index_pdf_bytes(filename, contents, config)
    graph_data = _fetch_graph(config)
    return {
        "filename": filename,
        "chunks_indexed": chunk_count,
        "graph": graph_data,
    }


def _index_pdf_bytes(filename: str, contents: bytes, config: RagConfig) -> int:
    safe_name = Path(filename).name or "upload.pdf"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / safe_name
        tmp_path.write_bytes(contents)
        pipeline = IndexingPipeline(config)
        try:
            return pipeline.run([tmp_path], overwrite=False)
        except Exception as exc:
            logger.exception("Indexing failed for upload")
            raise HTTPException(status_code=500, detail=f"Indexierung fehlgeschlagen: {exc}") from exc
        finally:
            pipeline.store.close()


def _fetch_graph(config: RagConfig) -> dict[str, Any]:
    store = Neo4jGraphStore(config.neo4j)
    try:
        return store.fetch_entity_graph()
    except Exception as exc:
        logger.exception("Graph fetch failed")
        raise HTTPException(status_code=500, detail=f"Graph konnte nicht geladen werden: {exc}") from exc
    finally:
        store.close()


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)

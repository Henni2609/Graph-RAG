from __future__ import annotations

from pathlib import Path

from kg_rag.compat import make_document
from kg_rag.components.entity_extractor import EntityExtractor
from kg_rag.config import LLMConfig, Neo4jConfig, RagConfig
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines import indexing as indexing_mod
from kg_rag.pipelines.indexing import IndexingPipeline


class _NoopStore:
    """Store stub: records what was called, no Neo4j contact."""

    def __init__(self) -> None:
        self.setup_called = False
        self.persisted: list = []

    def setup_schema(self, *, dimensions=None) -> None:
        self.setup_called = True

    def persist_documents(self, documents, *, overwrite=False, session_id="default", progress=None, batch_size=100):
        self.persisted = list(documents)
        if progress is not None:
            progress("persisting", 0, len(self.persisted))
            progress("persisting", len(self.persisted), len(self.persisted))

    def delete_stale_document_chunks(self, session_id, document_id, valid_chunk_ids):
        pass

    def delete_stale_chunks_bulk(self, doc_valid_ids, *, session_id):
        pass

    def store_indexing_meta(self, session_id, model, dimensions):
        pass

    def close(self) -> None:
        pass


class _NoopExtractor:
    def run(self, documents, *, progress=None):
        total = len(documents)
        if progress is not None:
            progress("extracting", 0, total)
            for i in range(total):
                progress("extracting", i + 1, total)
        enriched = []
        for doc in documents:
            doc.meta.setdefault("entities", [])
            doc.meta.setdefault("relations", [])
            enriched.append(doc)
        return {"documents": enriched}


def _build_config() -> RagConfig:
    return RagConfig(llm=LLMConfig(api_key="sk-test"), neo4j=Neo4jConfig())


def test_indexing_pipeline_emits_full_step_sequence(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        indexing_mod,
        "load_documents",
        lambda files, *, session_id="default": [
            make_document("paragraph one", meta={"source": str(pdf), "title": pdf.name, "session_id": session_id}),
        ],
    )
    monkeypatch.setattr(
        indexing_mod,
        "split_documents",
        lambda docs, *, split_length=10, split_overlap=2: [
            make_document(
                f"chunk-{i}",
                meta={"source": str(pdf), "title": pdf.name, "chunk_index": i},
            )
            for i in range(3)
        ],
    )
    monkeypatch.setattr(
        indexing_mod,
        "embed_documents",
        lambda docs, *, model, batch_size=64: docs,
    )

    pipeline = IndexingPipeline(_build_config(), store=_NoopStore(), entity_extractor=_NoopExtractor())

    events: list[tuple[str, int, int]] = []
    chunk_count = pipeline.run([pdf], progress=lambda step, c, t: events.append((step, c, t)))

    steps = [step for step, _c, _t in events]
    assert steps[0] == "parsing"
    assert "splitting" in steps
    assert "embedding" in steps
    assert "extracting" in steps
    assert "persisting" in steps
    assert steps[-1] == "done"
    assert events[-1] == ("done", 3, 3)
    assert chunk_count == 3


def test_indexing_pipeline_no_files_emits_done(tmp_path: Path) -> None:
    pipeline = IndexingPipeline(_build_config(), store=_NoopStore(), entity_extractor=_NoopExtractor())
    events: list[tuple[str, int, int]] = []
    count = pipeline.run([tmp_path / "missing.pdf"], progress=lambda s, c, t: events.append((s, c, t)))

    assert count == 0
    assert events == [("done", 0, 0)]

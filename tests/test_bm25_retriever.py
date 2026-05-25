from __future__ import annotations

import pytest

from kg_rag.components.bm25_retriever import BM25Retriever
from kg_rag.compat import document_meta


class FakeStore:
    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self.call_count = 0

    def execute_read(self, query: str, **kwargs: object) -> list[dict]:
        self.call_count += 1
        return self._records


SAMPLE_RECORDS = [
    {
        "id": "chunk-1",
        "text": "the quick brown fox",
        "chunk_index": 0,
        "page_number": 1,
        "document_id": "doc-1",
        "source": "test.pdf",
        "title": "Test",
        "section_title": None,
    },
    {
        "id": "chunk-2",
        "text": "jumps over the lazy dog",
        "chunk_index": 1,
        "page_number": 1,
        "document_id": "doc-1",
        "source": "test.pdf",
        "title": "Test",
        "section_title": None,
    },
    {
        "id": "chunk-3",
        "text": "something completely different",
        "chunk_index": 2,
        "page_number": 2,
        "document_id": "doc-1",
        "source": "test.pdf",
        "title": "Test",
        "section_title": None,
    },
]


@pytest.fixture(autouse=True)
def clear_bm25_cache():
    BM25Retriever._cache.clear()
    yield
    BM25Retriever._cache.clear()


def test_search_returns_bm25_documents():
    store = FakeStore(SAMPLE_RECORDS)
    retriever = BM25Retriever(store)
    results = retriever.search("quick fox", session_id="sess-1")
    assert len(results) > 0
    for doc in results:
        assert document_meta(doc).get("retrieval_source") == "bm25"


def test_cache_hit_does_not_call_execute_read_again():
    store = FakeStore(SAMPLE_RECORDS)
    retriever = BM25Retriever(store)
    retriever.search("quick fox", session_id="sess-1")
    assert store.call_count == 1
    retriever.search("lazy dog", session_id="sess-1")
    assert store.call_count == 1, "execute_read must not be called again on cache hit"


def test_invalidation_causes_rebuild():
    store = FakeStore(SAMPLE_RECORDS)
    retriever = BM25Retriever(store)
    retriever.search("quick fox", session_id="sess-1")
    assert store.call_count == 1
    BM25Retriever.invalidate("sess-1")
    retriever.search("lazy dog", session_id="sess-1")
    assert store.call_count == 2, "execute_read must be called again after invalidation"

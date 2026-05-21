from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from kg_rag.compat import Document, make_document

if TYPE_CHECKING:
    from kg_rag.neo4j_store import Neo4jGraphStore


class BM25Retriever:
    # Class-level cache so all instances share the same index and invalidation works globally.
    _cache: dict[str, tuple[Any, list[dict]]] = {}
    _lock = threading.Lock()

    def __init__(self, store: "Neo4jGraphStore") -> None:
        self.store = store

    @classmethod
    def invalidate(cls, session_id: str) -> None:
        with cls._lock:
            cls._cache.pop(session_id, None)

    def _build_index(self, session_id: str) -> tuple[Any, list[dict]]:
        from rank_bm25 import BM25Okapi

        records = self.store.execute_read(
            """
            MATCH (c:Chunk)
            WHERE c.session_id = $session_id AND c.text IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.chunk_index AS chunk_index,
                   c.page_number AS page_number, c.document_id AS document_id,
                   c.source AS source, c.title AS title, c.section_title AS section_title
            """,
            session_id=session_id,
        )
        records = list(records)
        if not records:
            return None, []
        tokenized = [r["text"].lower().split() for r in records]
        return BM25Okapi(tokenized), records

    def _get_index(self, session_id: str) -> tuple[Any, list[dict]]:
        with self.__class__._lock:
            if session_id not in self.__class__._cache:
                self.__class__._cache[session_id] = self._build_index(session_id)
            return self.__class__._cache[session_id]

    def search(self, query: str, *, session_id: str, top_k: int = 60) -> list[Document]:
        try:
            bm25, records = self._get_index(session_id)
        except Exception:
            return []
        if bm25 is None or not records:
            return []
        tokenized_query = query.lower().split()
        raw = bm25.get_scores(tokenized_query)
        scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        result = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            rec = records[idx]
            try:
                page = max(1, int(rec.get("page_number") or 1))
            except (TypeError, ValueError):
                page = 1
            doc = make_document(
                rec.get("text", "") or "",
                meta={
                    "chunk_id": rec.get("id"),
                    "chunk_index": rec.get("chunk_index"),
                    "page_number": page,
                    "document_id": rec.get("document_id"),
                    "source": rec.get("source"),
                    "title": rec.get("title"),
                    "section_title": rec.get("section_title"),
                    "retrieval_source": "bm25",
                },
                score=float(scores[idx]),
            )
            result.append(doc)
        return result

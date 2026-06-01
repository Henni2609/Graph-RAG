from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from kg_rag.compat import Document, make_document

if TYPE_CHECKING:
    from kg_rag.neo4j_store import Neo4jGraphStore


# Word-boundary tokenisation: keeps hyphenated/punctuated tokens like
# "Graph-RAG" matchable against "graph rag" and ignores stray punctuation.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    # Class-level cache so all instances share the same index and invalidation works globally.
    _MAX_CACHED_SESSIONS = 32
    _cache: OrderedDict[str, tuple[Any, list[dict]]] = OrderedDict()
    _versions: dict[str, int] = {}
    _lock = threading.Lock()

    def __init__(self, store: "Neo4jGraphStore") -> None:
        self.store = store

    @classmethod
    def invalidate(cls, session_id: str) -> None:
        with cls._lock:
            cls._cache.pop(session_id, None)
            cls._versions[session_id] = cls._versions.get(session_id, 0) + 1

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
        tokenized = [_tokenize(r["text"]) for r in records]
        return BM25Okapi(tokenized), records

    def _get_index(self, session_id: str) -> tuple[Any, list[dict]]:
        cls = self.__class__
        with cls._lock:
            v_before = cls._versions.get(session_id, 0)
            cached = cls._cache.get(session_id)
            if cached is not None:
                cls._cache.move_to_end(session_id)
                return cached
        built = self._build_index(session_id)
        with cls._lock:
            if cls._versions.get(session_id, 0) != v_before:
                # Invalidation happened during build — don't cache a stale index.
                return built
            # Cache empty results too — avoids re-running the Neo4j query on every
            # search against an empty/not-yet-indexed session.
            cls._cache[session_id] = built
            cls._cache.move_to_end(session_id)
            while len(cls._cache) > cls._MAX_CACHED_SESSIONS:
                cls._cache.popitem(last=False)
        return built

    def search(self, query: str, *, session_id: str, top_k: int = 60) -> list[Document]:
        try:
            bm25, records = self._get_index(session_id)
        except Exception:
            return []
        if bm25 is None or not records:
            return []
        tokenized_query = _tokenize(query)
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

from __future__ import annotations

from typing import Any

from kg_rag.compat import Document, component
from kg_rag.config import Neo4jConfig
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore


@component
class GraphRetriever:
    def __init__(
        self,
        store: Neo4jGraphStore | None = None,
        neo4j_config: Neo4jConfig | None = None,
        hops: int = 2,
        limit: int = 8,
    ) -> None:
        self.store = store
        self.neo4j_config = neo4j_config
        self.hops = max(1, min(3, hops))
        self.limit = limit

    @component.output_types(documents=list[Document], entity_context=str)
    def run(
        self,
        chunk_ids: list[str],
        query_entities: list[str] | None = None,
        hops: int | None = None,
        limit: int | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> dict[str, Any]:
        store = self._store()
        active_hops = max(1, min(3, hops if hops is not None else self.hops))
        active_limit = limit if limit is not None else self.limit
        documents = store.graph_search(
            chunk_ids=chunk_ids,
            query_entities=query_entities or [],
            hops=active_hops,
            limit=active_limit,
            session_id=session_id,
        )
        entity_context = store.entity_context(query_entities or [], limit=30, session_id=session_id)
        return {"documents": documents, "entity_context": entity_context}

    def _store(self) -> Neo4jGraphStore:
        if self.store is not None:
            return self.store
        if self.neo4j_config is None:
            self.neo4j_config = Neo4jConfig.from_env()
        self.store = Neo4jGraphStore(self.neo4j_config)
        return self.store

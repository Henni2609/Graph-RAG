from __future__ import annotations

import functools
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from kg_rag.compat import Document, document_content, document_embedding, document_meta, make_document
from kg_rag.config import Neo4jConfig
from kg_rag.logging import logger
from kg_rag.schema import normalize_entity_name


VECTOR_INDEX_NAME = "chunk_embeddings"
_DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_SESSION_ID = "default"


@functools.lru_cache(maxsize=None)
def _cached_driver(uri: str, username: str, password: str) -> Any:
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=(username, password))


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    embedding: list[float] | None
    chunk_index: int
    document_id: str
    source: str
    title: str
    session_id: str
    page_number: int = 1


class Neo4jGraphStore:
    def __init__(self, config: Neo4jConfig, driver: Any | None = None) -> None:
        self.config = config
        self._driver = driver
        self._injected_driver = driver is not None

    @property
    def driver(self) -> Any:
        if self._driver is None:
            self._driver = _cached_driver(
                self.config.uri, self.config.username, self.config.password
            )
        return self._driver

    def close(self) -> None:
        # Only close explicitly-injected drivers; cached drivers live for the process.
        if self._injected_driver and self._driver is not None:
            self._driver.close()

    def clear(self) -> None:
        self.execute_write("MATCH (n) DETACH DELETE n")

    def delete_session(self, session_id: str) -> None:
        if not session_id:
            return
        self.execute_write(
            "MATCH (n) WHERE n.session_id = $session_id DETACH DELETE n",
            session_id=session_id,
        )

    def delete_orphan_chunks(self, session_id: str, valid_document_ids: set[str]) -> None:
        if not session_id or not valid_document_ids:
            return
        valid_ids = list(valid_document_ids)
        # Cheap read: skip all writes when nothing to clean up (common case).
        has_orphans = self.execute_read(
            "MATCH (c:Chunk {session_id: $session_id}) "
            "WHERE NOT c.document_id IN $valid_ids "
            "RETURN true AS has_orphans LIMIT 1",
            session_id=session_id,
            valid_ids=valid_ids,
        )
        if not has_orphans:
            return
        self.execute_write(
            "MATCH (c:Chunk {session_id: $session_id}) "
            "WHERE NOT c.document_id IN $valid_ids "
            "DETACH DELETE c",
            session_id=session_id,
            valid_ids=valid_ids,
        )
        self.execute_write(
            "MATCH (d:Document {session_id: $session_id}) "
            "WHERE NOT d.id IN $valid_ids "
            "DETACH DELETE d",
            session_id=session_id,
            valid_ids=valid_ids,
        )
        self.execute_write(
            "MATCH (e:Entity {session_id: $session_id}) "
            "WHERE NOT (e)<-[:MENTIONS]-(:Chunk) "
            "DETACH DELETE e",
            session_id=session_id,
        )

    def delete_stale_document_chunks(
        self, session_id: str, document_id: str, valid_chunk_ids: set[str]
    ) -> None:
        """Remove chunks of a document that no longer exist after re-indexing."""
        if not valid_chunk_ids:
            return
        self.execute_write(
            "MATCH (c:Chunk {document_id: $document_id, session_id: $session_id}) "
            "WHERE NOT c.id IN $valid_ids "
            "DETACH DELETE c",
            document_id=document_id,
            session_id=session_id,
            valid_ids=list(valid_chunk_ids),
        )

    def delete_stale_chunks_bulk(
        self, doc_valid_ids: dict[str, list[str]], *, session_id: str
    ) -> None:
        payload = [
            {"doc_id": doc_id, "valid_ids": valid_ids}
            for doc_id, valid_ids in doc_valid_ids.items()
            if valid_ids
        ]
        if not payload:
            return
        self.execute_write(
            """
            UNWIND $payload AS row
            MATCH (c:Chunk {document_id: row.doc_id, session_id: $session_id})
            WHERE NOT c.id IN row.valid_ids
            DETACH DELETE c
            """,
            payload=payload,
            session_id=session_id,
        )

    def setup_schema(self, *, dimensions: int | None = None) -> None:
        effective_dim = dimensions if dimensions is not None else _DEFAULT_EMBEDDING_DIMENSIONS
        statements = [
            "DROP CONSTRAINT entity_name_normalized IF EXISTS",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            (
                "CREATE CONSTRAINT entity_name_session IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.name_normalized, e.session_id) IS UNIQUE"
            ),
            (
                f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {effective_dim}, "
                "`vector.similarity_function`: 'cosine'"
                "}}"
            ),
        ]
        for statement in statements:
            self.execute_write(statement)

    def store_indexing_meta(self, session_id: str, model: str, dimensions: int) -> None:
        """Persist the embedding model name and dimensions used for this session."""
        self.execute_write(
            """
            MERGE (m:IndexingMeta {session_id: $session_id})
            SET m.embedding_model = $model, m.embedding_dimensions = $dimensions
            """,
            session_id=session_id,
            model=model,
            dimensions=dimensions,
        )

    def get_indexing_meta(self, session_id: str) -> dict[str, Any] | None:
        """Return stored embedding model and dimensions for this session, or None."""
        records = self.execute_read(
            "MATCH (m:IndexingMeta {session_id: $session_id}) "
            "RETURN m.embedding_model AS model, m.embedding_dimensions AS dimensions",
            session_id=session_id,
        )
        return dict(records[0]) if records else None

    def persist_documents(
        self,
        documents: list[Document],
        *,
        overwrite: bool = False,
        session_id: str = DEFAULT_SESSION_ID,
        progress: Callable[[str, int, int], None] | None = None,
        batch_size: int = 100,
    ) -> None:
        if overwrite:
            self.clear()
            self.setup_schema()

        total_chunks = len(documents)
        if progress is not None:
            progress("persisting", 0, total_chunks)
        if not documents:
            return

        stored_chunks: list[StoredChunk] = []
        chunks_by_document: dict[str, list[StoredChunk]] = defaultdict(list)
        chunk_metas: dict[str, dict[str, Any]] = {}
        for document in documents:
            chunk = stored_chunk_from_document(document, session_id=session_id)
            stored_chunks.append(chunk)
            chunks_by_document[chunk.document_id].append(chunk)
            chunk_metas[chunk.id] = document_meta(document)

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1) Batch: Documents + Chunks + HAS_CHUNK
        done = 0
        for batch in _chunks_of(stored_chunks, batch_size):
            payload = [
                {
                    "document_id": c.document_id,
                    "title": c.title,
                    "source": c.source,
                    "chunk_id": c.id,
                    "text": c.text,
                    "embedding": c.embedding,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                }
                for c in batch
            ]
            self.execute_write(
                """
                UNWIND $chunks AS c
                MERGE (d:Document {id: c.document_id})
                  ON CREATE SET d.created_at = $created_at
                SET d.title = c.title,
                    d.source = c.source,
                    d.session_id = $session_id,
                    d.updated_at = $created_at
                MERGE (chunk:Chunk {id: c.chunk_id})
                SET chunk.text = c.text,
                    chunk.embedding = c.embedding,
                    chunk.chunk_index = c.chunk_index,
                    chunk.page_number = c.page_number,
                    chunk.document_id = c.document_id,
                    chunk.source = c.source,
                    chunk.title = c.title,
                    chunk.session_id = $session_id
                MERGE (d)-[:HAS_CHUNK]->(chunk)
                """,
                chunks=payload,
                session_id=session_id,
                created_at=now_iso,
            )
            done += len(batch)
            if progress is not None:
                progress("persisting", done, total_chunks)

        # 2) Batch: Entities + MENTIONS
        entities_payload: list[dict[str, Any]] = []
        for chunk_id, meta in chunk_metas.items():
            for entity in meta.get("entities") or []:
                name = str(entity.get("name", "")).strip()
                if not name:
                    continue
                entities_payload.append(
                    {
                        "chunk_id": chunk_id,
                        "name": name,
                        "name_normalized": normalize_entity_name(name),
                        "type": str(entity.get("type", "Konzept") or "Konzept"),
                        "description": str(entity.get("description", "") or ""),
                    }
                )
        for batch in _chunks_of(entities_payload, batch_size * 10):
            self.execute_write(
                """
                UNWIND $entities AS row
                MERGE (e:Entity {name_normalized: row.name_normalized, session_id: $session_id})
                SET e.name = row.name,
                    e.type = row.type,
                    e.description = row.description
                WITH e, row
                MATCH (c:Chunk {id: row.chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                entities=batch,
                session_id=session_id,
            )

        # 3) Batch: RELATES_TO — MERGE endpoints so missing entities are auto-created;
        #    key by (source, target, relation) only to avoid duplicate parallel edges.
        relations_payload: list[dict[str, Any]] = []
        for chunk_id, meta in chunk_metas.items():
            for relation in meta.get("relations") or []:
                source_norm = normalize_entity_name(str(relation.get("source", "")))
                target_norm = normalize_entity_name(str(relation.get("target", "")))
                relation_type = str(relation.get("relation", "RELATES_TO") or "RELATES_TO")
                if not source_norm or not target_norm:
                    continue
                relations_payload.append(
                    {
                        "source": source_norm,
                        "source_name": str(relation.get("source", "")).strip(),
                        "target": target_norm,
                        "target_name": str(relation.get("target", "")).strip(),
                        "relation": relation_type,
                        "chunk_id": chunk_id,
                    }
                )
        for batch in _chunks_of(relations_payload, batch_size * 10):
            self.execute_write(
                """
                UNWIND $relations AS row
                MERGE (src:Entity {name_normalized: row.source, session_id: $session_id})
                  ON CREATE SET src.name = row.source_name, src.type = 'Konzept'
                MERGE (tgt:Entity {name_normalized: row.target, session_id: $session_id})
                  ON CREATE SET tgt.name = row.target_name, tgt.type = 'Konzept'
                MERGE (src)-[rel:RELATES_TO {relation: row.relation}]->(tgt)
                  ON CREATE SET rel.chunk_ids = [row.chunk_id]
                  ON MATCH SET rel.chunk_ids = CASE
                    WHEN row.chunk_id IN rel.chunk_ids THEN rel.chunk_ids
                    ELSE rel.chunk_ids + [row.chunk_id]
                  END
                """,
                relations=batch,
                session_id=session_id,
            )

        # 4) Batch: NEXT_CHUNK edges
        next_pairs: list[dict[str, str]] = []
        for chunks in chunks_by_document.values():
            ordered = sorted(chunks, key=lambda item: item.chunk_index)
            for previous, current in zip(ordered, ordered[1:]):
                next_pairs.append({"previous_id": previous.id, "current_id": current.id})
        for batch in _chunks_of(next_pairs, batch_size * 10):
            self.execute_write(
                """
                UNWIND $pairs AS pair
                MATCH (previous:Chunk {id: pair.previous_id})
                MATCH (current:Chunk {id: pair.current_id})
                MERGE (previous)-[:NEXT_CHUNK]->(current)
                """,
                pairs=batch,
            )

        if progress is not None:
            progress("persisting", total_chunks, total_chunks)

    def vector_search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[Document]:
        # Session-scoped exact cosine: avoids cross-session crowding that breaks
        # the global-queryNodes+post-filter pattern when multiple sessions share
        # one index. vector.similarity.cosine() available since Neo4j 5.18.
        records = self.execute_read(
            """
            MATCH (c:Chunk)
            WHERE c.session_id = $session_id AND c.embedding IS NOT NULL
            WITH c, vector.similarity.cosine(c.embedding, $embedding) AS score
            RETURN c.id AS id,
                   c.text AS text,
                   c.chunk_index AS chunk_index,
                   c.page_number AS page_number,
                   c.document_id AS document_id,
                   c.source AS source,
                   c.title AS title,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            top_k=top_k,
            embedding=embedding,
            session_id=session_id,
        )
        return [document_from_record(record, source_label="vector") for record in records]

    def graph_search(
        self,
        *,
        chunk_ids: list[str],
        query_embedding: list[float] | None = None,
        query_entities: list[str] | None = None,
        hops: int = 2,
        limit: int = 8,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[Document]:
        hops = max(1, min(3, hops))
        normalized_entities = [normalize_entity_name(name) for name in query_entities or [] if name]
        params: dict[str, Any] = {
            "chunk_ids": chunk_ids,
            "query_entities": normalized_entities,
            "limit": limit,
            "session_id": session_id,
        }
        if query_embedding:
            params["query_embedding"] = query_embedding
            score_expr = "COALESCE(vector.similarity.cosine(chunk.embedding, $query_embedding), 0.0)"
        else:
            score_expr = "1.0"
        records = self.execute_read(
            f"""
            MATCH (seed:Chunk)
            WHERE seed.id IN $chunk_ids AND seed.session_id = $session_id
            MATCH (seed)-[:MENTIONS]->(:Entity)-[:RELATES_TO*1..{hops}]-(:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE NOT chunk.id IN $chunk_ids AND chunk.session_id = $session_id
            RETURN DISTINCT chunk.id AS id,
                   chunk.text AS text,
                   chunk.chunk_index AS chunk_index,
                   chunk.page_number AS page_number,
                   chunk.document_id AS document_id,
                   chunk.source AS source,
                   chunk.title AS title,
                   {score_expr} AS score
            UNION
            MATCH (query_entity:Entity)
            WHERE query_entity.name_normalized IN $query_entities
              AND query_entity.session_id = $session_id
            MATCH (query_entity)-[:RELATES_TO*0..{hops}]-(:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE NOT chunk.id IN $chunk_ids AND chunk.session_id = $session_id
            RETURN DISTINCT chunk.id AS id,
                   chunk.text AS text,
                   chunk.chunk_index AS chunk_index,
                   chunk.page_number AS page_number,
                   chunk.document_id AS document_id,
                   chunk.source AS source,
                   chunk.title AS title,
                   {score_expr} AS score
            LIMIT $limit
            """,
            **params,
        )
        return [document_from_record(record, source_label="graph") for record in records]

    def fetch_entity_graph(
        self,
        *,
        limit: int = 500,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> dict[str, list[dict[str, Any]]]:
        node_records = self.execute_read(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
            RETURN e.name_normalized AS id,
                   coalesce(e.name, e.name_normalized) AS label,
                   coalesce(e.type, 'Konzept') AS type,
                   coalesce(e.description, '') AS description
            LIMIT $limit
            """,
            limit=limit,
            session_id=session_id,
        )
        edge_records = self.execute_read(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE a.session_id = $session_id AND b.session_id = $session_id
            RETURN a.name_normalized AS source,
                   b.name_normalized AS target,
                   coalesce(r.relation, 'RELATES_TO') AS relation
            LIMIT $edge_limit
            """,
            edge_limit=limit * 4,
            session_id=session_id,
        )
        return {
            "nodes": [dict(record) for record in node_records],
            "edges": [dict(record) for record in edge_records],
        }

    def entity_context(
        self,
        entity_names: list[str],
        *,
        limit: int = 30,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> str:
        normalized_names = [normalize_entity_name(name) for name in entity_names if name]
        if not normalized_names:
            return ""

        records = self.execute_read(
            """
            MATCH (source:Entity)
            WHERE source.name_normalized IN $names AND source.session_id = $session_id
            MATCH (source)-[rel:RELATES_TO]-(target:Entity)
            WHERE target.session_id = $session_id
            RETURN DISTINCT source.name AS source,
                   coalesce(rel.relation, 'RELATES_TO') AS relation,
                   target.name AS target,
                   coalesce(source.description, '') AS source_description,
                   coalesce(target.description, '') AS target_description
            LIMIT $limit
            """,
            names=normalized_names,
            limit=limit,
            session_id=session_id,
        )
        lines = []
        for record in records:
            lines.append(f"- {record['source']} --{record['relation']}--> {record['target']}")
        return "\n".join(lines)

    def execute_write(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, **parameters)
            return [dict(record) for record in result]

    def execute_read(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, **parameters)
            return [dict(record) for record in result]


def _chunks_of(items: list[Any], size: int) -> Iterator[list[Any]]:
    if not items:
        return
    step = max(1, size)
    for i in range(0, len(items), step):
        yield items[i : i + step]


def stored_chunk_from_document(
    document: Document,
    *,
    session_id: str = DEFAULT_SESSION_ID,
) -> StoredChunk:
    meta = document_meta(document)
    source = str(meta.get("source", "unknown"))
    title = str(meta.get("title") or Path(source).name or "Untitled")
    document_id = str(meta.get("document_id") or stable_id(f"{session_id}|{source}"))
    chunk_index = int(meta.get("chunk_index", meta.get("split_idx", 0)) or 0)
    try:
        page_number = int(meta.get("page_number") or 1)
    except (TypeError, ValueError):
        page_number = 1
    chunk_id = str(meta.get("chunk_id") or stable_id(f"{session_id}|{document_id}|{chunk_index}"))
    return StoredChunk(
        id=chunk_id,
        text=document_content(document),
        embedding=document_embedding(document),
        chunk_index=chunk_index,
        document_id=document_id,
        source=source,
        title=title,
        session_id=session_id,
        page_number=max(1, page_number),
    )


def document_from_record(record: dict[str, Any], *, source_label: str) -> Document:
    page_number = record.get("page_number")
    try:
        page_number = int(page_number) if page_number is not None else 1
    except (TypeError, ValueError):
        page_number = 1
    return make_document(
        record.get("text", "") or "",
        meta={
            "chunk_id": record.get("id"),
            "chunk_index": record.get("chunk_index"),
            "page_number": max(1, page_number),
            "document_id": record.get("document_id"),
            "source": record.get("source"),
            "title": record.get("title"),
            "retrieval_source": source_label,
        },
        score=record.get("score"),
    )


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

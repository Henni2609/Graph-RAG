from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from kg_rag.compat import Document, document_content, document_embedding, document_meta, make_document
from kg_rag.config import Neo4jConfig
from kg_rag.schema import normalize_entity_name


VECTOR_INDEX_NAME = "chunk_embeddings"
EMBEDDING_DIMENSIONS = 384
DEFAULT_SESSION_ID = "default"


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

    @property
    def driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
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

    def setup_schema(self) -> None:
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
                f"`vector.dimensions`: {EMBEDDING_DIMENSIONS}, "
                "`vector.similarity_function`: 'cosine'"
                "}}"
            ),
        ]
        for statement in statements:
            self.execute_write(statement)

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

        # 3) Batch: RELATES_TO between entities
        relations_payload: list[dict[str, Any]] = []
        for chunk_id, meta in chunk_metas.items():
            for relation in meta.get("relations") or []:
                source = normalize_entity_name(str(relation.get("source", "")))
                target = normalize_entity_name(str(relation.get("target", "")))
                relation_type = str(relation.get("relation", "RELATES_TO") or "RELATES_TO")
                if not source or not target:
                    continue
                relations_payload.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": relation_type,
                        "chunk_id": chunk_id,
                    }
                )
        for batch in _chunks_of(relations_payload, batch_size * 10):
            self.execute_write(
                """
                UNWIND $relations AS row
                MATCH (source:Entity {name_normalized: row.source, session_id: $session_id})
                MATCH (target:Entity {name_normalized: row.target, session_id: $session_id})
                MERGE (source)-[rel:RELATES_TO {relation: row.relation, chunk_id: row.chunk_id}]->(target)
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
        oversample = max(top_k * 5, top_k + 10)
        records = self.execute_read(
            f"""
            CALL db.index.vector.queryNodes($index_name, $oversample, $embedding)
            YIELD node, score
            WHERE node.session_id = $session_id
            RETURN node.id AS id,
                   node.text AS text,
                   node.chunk_index AS chunk_index,
                   node.page_number AS page_number,
                   node.document_id AS document_id,
                   node.source AS source,
                   node.title AS title,
                   score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            index_name=VECTOR_INDEX_NAME,
            oversample=oversample,
            top_k=top_k,
            embedding=embedding,
            session_id=session_id,
        )
        return [document_from_record(record, source_label="vector") for record in records]

    def graph_search(
        self,
        *,
        chunk_ids: list[str],
        query_entities: list[str] | None = None,
        hops: int = 2,
        limit: int = 8,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[Document]:
        hops = max(1, min(3, hops))
        normalized_entities = [normalize_entity_name(name) for name in query_entities or [] if name]
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
                   1.0 AS score
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
                   1.0 AS score
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            query_entities=normalized_entities,
            limit=limit,
            session_id=session_id,
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
    chunk_id = str(
        meta.get("chunk_id")
        or stable_id(f"{session_id}|{document_id}|{chunk_index}|{document_content(document)}")
    )
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

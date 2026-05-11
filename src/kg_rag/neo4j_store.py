from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kg_rag.compat import Document, document_content, document_embedding, document_meta, make_document
from kg_rag.config import Neo4jConfig
from kg_rag.schema import normalize_entity_name


VECTOR_INDEX_NAME = "chunk_embeddings"
EMBEDDING_DIMENSIONS = 384


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    embedding: list[float] | None
    chunk_index: int
    document_id: str
    source: str
    title: str


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

    def setup_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            (
                "CREATE CONSTRAINT entity_name_normalized IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name_normalized IS UNIQUE"
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

    def persist_documents(self, documents: list[Document], *, overwrite: bool = False) -> None:
        if overwrite:
            self.clear()
            self.setup_schema()

        chunks_by_document: dict[str, list[StoredChunk]] = defaultdict(list)

        for document in documents:
            meta = document_meta(document)
            chunk = stored_chunk_from_document(document)
            chunks_by_document[chunk.document_id].append(chunk)
            self.execute_write(
                """
                MERGE (d:Document {id: $document_id})
                SET d.title = $title,
                    d.source = $source,
                    d.created_at = coalesce(d.created_at, $created_at),
                    d.updated_at = $created_at
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.embedding = $embedding,
                    c.chunk_index = $chunk_index,
                    c.document_id = $document_id,
                    c.source = $source,
                    c.title = $title
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                created_at=datetime.now(timezone.utc).isoformat(),
                chunk_id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
                chunk_index=chunk.chunk_index,
            )
            self._persist_entities_and_relations(chunk.id, meta)

        for chunks in chunks_by_document.values():
            ordered = sorted(chunks, key=lambda item: item.chunk_index)
            for previous, current in zip(ordered, ordered[1:]):
                self.execute_write(
                    """
                    MATCH (previous:Chunk {id: $previous_id})
                    MATCH (current:Chunk {id: $current_id})
                    MERGE (previous)-[:NEXT_CHUNK]->(current)
                    """,
                    previous_id=previous.id,
                    current_id=current.id,
                )

    def vector_search(self, embedding: list[float], *, top_k: int = 5) -> list[Document]:
        records = self.execute_read(
            f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            RETURN node.id AS id,
                   node.text AS text,
                   node.chunk_index AS chunk_index,
                   node.document_id AS document_id,
                   node.source AS source,
                   node.title AS title,
                   score
            ORDER BY score DESC
            """,
            index_name=VECTOR_INDEX_NAME,
            top_k=top_k,
            embedding=embedding,
        )
        return [document_from_record(record, source_label="vector") for record in records]

    def graph_search(
        self,
        *,
        chunk_ids: list[str],
        query_entities: list[str] | None = None,
        hops: int = 2,
        limit: int = 8,
    ) -> list[Document]:
        hops = max(1, min(3, hops))
        normalized_entities = [normalize_entity_name(name) for name in query_entities or [] if name]
        records = self.execute_read(
            f"""
            MATCH (seed:Chunk)
            WHERE seed.id IN $chunk_ids
            MATCH (seed)-[:MENTIONS]->(:Entity)-[:RELATES_TO*1..{hops}]-(:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE NOT chunk.id IN $chunk_ids
            RETURN DISTINCT chunk.id AS id,
                   chunk.text AS text,
                   chunk.chunk_index AS chunk_index,
                   chunk.document_id AS document_id,
                   chunk.source AS source,
                   chunk.title AS title,
                   1.0 AS score
            UNION
            MATCH (query_entity:Entity)
            WHERE query_entity.name_normalized IN $query_entities
            MATCH (query_entity)-[:RELATES_TO*0..{hops}]-(:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE NOT chunk.id IN $chunk_ids
            RETURN DISTINCT chunk.id AS id,
                   chunk.text AS text,
                   chunk.chunk_index AS chunk_index,
                   chunk.document_id AS document_id,
                   chunk.source AS source,
                   chunk.title AS title,
                   1.0 AS score
            LIMIT $limit
            """,
            chunk_ids=chunk_ids,
            query_entities=normalized_entities,
            limit=limit,
        )
        return [document_from_record(record, source_label="graph") for record in records]

    def fetch_entity_graph(self, *, limit: int = 500) -> dict[str, list[dict[str, Any]]]:
        node_records = self.execute_read(
            """
            MATCH (e:Entity)
            RETURN e.name_normalized AS id,
                   coalesce(e.name, e.name_normalized) AS label,
                   coalesce(e.type, 'Konzept') AS type,
                   coalesce(e.description, '') AS description
            LIMIT $limit
            """,
            limit=limit,
        )
        edge_records = self.execute_read(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.name_normalized AS source,
                   b.name_normalized AS target,
                   coalesce(r.relation, 'RELATES_TO') AS relation
            LIMIT $edge_limit
            """,
            edge_limit=limit * 4,
        )
        return {
            "nodes": [dict(record) for record in node_records],
            "edges": [dict(record) for record in edge_records],
        }

    def entity_context(self, entity_names: list[str], *, limit: int = 30) -> str:
        normalized_names = [normalize_entity_name(name) for name in entity_names if name]
        if not normalized_names:
            return ""

        records = self.execute_read(
            """
            MATCH (source:Entity)
            WHERE source.name_normalized IN $names
            MATCH (source)-[rel:RELATES_TO]-(target:Entity)
            RETURN DISTINCT source.name AS source,
                   coalesce(rel.relation, 'RELATES_TO') AS relation,
                   target.name AS target,
                   coalesce(source.description, '') AS source_description,
                   coalesce(target.description, '') AS target_description
            LIMIT $limit
            """,
            names=normalized_names,
            limit=limit,
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

    def _persist_entities_and_relations(self, chunk_id: str, meta: dict[str, Any]) -> None:
        entities = meta.get("entities", []) or []
        relations = meta.get("relations", []) or []

        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            self.execute_write(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (e:Entity {name_normalized: $name_normalized})
                SET e.name = $name,
                    e.type = $type,
                    e.description = $description
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                name=name,
                name_normalized=normalize_entity_name(name),
                type=str(entity.get("type", "Konzept") or "Konzept"),
                description=str(entity.get("description", "") or ""),
            )

        for relation in relations:
            source = normalize_entity_name(str(relation.get("source", "")))
            target = normalize_entity_name(str(relation.get("target", "")))
            relation_type = str(relation.get("relation", "RELATES_TO") or "RELATES_TO")
            if not source or not target:
                continue
            self.execute_write(
                """
                MATCH (source:Entity {name_normalized: $source})
                MATCH (target:Entity {name_normalized: $target})
                MERGE (source)-[rel:RELATES_TO {relation: $relation, chunk_id: $chunk_id}]->(target)
                """,
                source=source,
                target=target,
                relation=relation_type,
                chunk_id=chunk_id,
            )


def stored_chunk_from_document(document: Document) -> StoredChunk:
    meta = document_meta(document)
    source = str(meta.get("source", "unknown"))
    title = str(meta.get("title") or Path(source).name or "Untitled")
    document_id = str(meta.get("document_id") or stable_id(source))
    chunk_index = int(meta.get("chunk_index", meta.get("split_idx", 0)) or 0)
    chunk_id = str(meta.get("chunk_id") or stable_id(f"{document_id}:{chunk_index}:{document_content(document)}"))
    return StoredChunk(
        id=chunk_id,
        text=document_content(document),
        embedding=document_embedding(document),
        chunk_index=chunk_index,
        document_id=document_id,
        source=source,
        title=title,
    )


def document_from_record(record: dict[str, Any], *, source_label: str) -> Document:
    return make_document(
        record.get("text", "") or "",
        meta={
            "chunk_id": record.get("id"),
            "chunk_index": record.get("chunk_index"),
            "document_id": record.get("document_id"),
            "source": record.get("source"),
            "title": record.get("title"),
            "retrieval_source": source_label,
        },
        score=record.get("score"),
    )


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

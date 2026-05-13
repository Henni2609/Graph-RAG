from kg_rag.compat import make_document
from kg_rag.config import Neo4jConfig
from kg_rag.neo4j_store import Neo4jGraphStore


class FakeResult:
    def __iter__(self):
        return iter([])


class FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self, database):
        self.calls.append(("database", database))
        return FakeSession(self.calls)

    def close(self):
        pass


def test_persist_documents_writes_chunks_entities_relations_and_next_edges() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(Neo4jConfig(), driver=driver)
    first = make_document(
        "Haystack uses Neo4j.",
        meta={
            "source": "/tmp/doc.md",
            "title": "doc.md",
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "entities": [
                {"name": "Haystack", "type": "Technologie", "description": ""},
                {"name": "Neo4j", "type": "Technologie", "description": ""},
            ],
            "relations": [
                {"source": "Haystack", "target": "Neo4j", "relation": "VERWENDET"},
            ],
        },
        embedding=[0.1] * 384,
    )
    second = make_document(
        "Neo4j stores graph relations.",
        meta={
            "source": "/tmp/doc.md",
            "title": "doc.md",
            "document_id": "doc-1",
            "chunk_id": "chunk-2",
            "chunk_index": 1,
        },
        embedding=[0.2] * 384,
    )

    store.persist_documents([first, second])

    cypher_calls = [query for query, _params in driver.calls if query != "database"]
    queries = "\n".join(cypher_calls)
    assert "UNWIND $chunks AS c" in queries
    assert "MERGE (chunk:Chunk {id: c.chunk_id})" in queries
    assert "UNWIND $entities AS row" in queries
    assert "MERGE (c)-[:MENTIONS]->(e)" in queries
    assert "UNWIND $relations AS row" in queries
    assert "MERGE (source)-[rel:RELATES_TO" in queries
    assert "UNWIND $pairs AS pair" in queries
    assert "MERGE (previous)-[:NEXT_CHUNK]->(current)" in queries
    # Genau vier UNWIND-Batches (chunks, entities, relations, next-pairs).
    # Pro Pipeline-Run, nicht pro Chunk.
    assert len(cypher_calls) == 4


def test_persist_documents_invokes_progress_callback() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(Neo4jConfig(), driver=driver)
    documents = [
        make_document(
            f"text-{i}",
            meta={
                "source": "/tmp/doc.md",
                "title": "doc.md",
                "document_id": "doc-1",
                "chunk_id": f"chunk-{i}",
                "chunk_index": i,
            },
            embedding=[0.1] * 384,
        )
        for i in range(5)
    ]
    events: list[tuple[str, int, int]] = []
    store.persist_documents(documents, progress=lambda step, c, t: events.append((step, c, t)))

    assert events[0] == ("persisting", 0, 5)
    assert events[-1] == ("persisting", 5, 5)
    # Monoton steigend
    counters = [c for _step, c, _t in events]
    assert counters == sorted(counters)


def test_graph_search_clamps_hops_to_three() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(Neo4jConfig(), driver=driver)

    store.graph_search(chunk_ids=["chunk-1"], query_entities=["Neo4j"], hops=99, limit=3)

    graph_query = next(query for query, params in driver.calls if "RELATES_TO*1..3" in query)
    assert "RELATES_TO*0..3" in graph_query

from __future__ import annotations

from fastapi.testclient import TestClient

from kg_rag.compat import make_document
from kg_rag.config import HuggingFaceConfig, Neo4jConfig, RagConfig
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines.query import QueryResult
from kg_rag.web.app import create_app


class FakeRecord(dict):
    pass


class FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class FakeSession:
    def __init__(self, results_by_marker):
        self.results_by_marker = results_by_marker

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, **_parameters):
        if "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)" in query:
            return FakeResult(self.results_by_marker["edges"])
        if "MATCH (e:Entity)" in query:
            return FakeResult(self.results_by_marker["nodes"])
        return FakeResult([])


class FakeDriver:
    def __init__(self, nodes, edges):
        self.results = {"nodes": nodes, "edges": edges}

    def session(self, database):
        return FakeSession(self.results)

    def close(self):
        pass


def _build_config() -> RagConfig:
    return RagConfig(
        hf=HuggingFaceConfig(api_token="hf_test"),
        neo4j=Neo4jConfig(),
    )


def test_fetch_entity_graph_returns_nodes_and_edges() -> None:
    nodes = [
        FakeRecord(id="haystack", label="Haystack", type="Technologie", description=""),
        FakeRecord(id="neo4j", label="Neo4j", type="Technologie", description="Graph DB"),
    ]
    edges = [FakeRecord(source="haystack", target="neo4j", relation="VERWENDET")]
    store = Neo4jGraphStore(Neo4jConfig(), driver=FakeDriver(nodes, edges))

    graph = store.fetch_entity_graph()

    assert graph["nodes"] == [dict(node) for node in nodes]
    assert graph["edges"] == [dict(edge) for edge in edges]


def test_graph_endpoint_returns_graph_payload(monkeypatch) -> None:
    nodes = [FakeRecord(id="haystack", label="Haystack", type="Technologie", description="")]
    edges = [FakeRecord(source="haystack", target="neo4j", relation="VERWENDET")]
    driver = FakeDriver(nodes, edges)

    monkeypatch.setattr(
        "kg_rag.web.app.Neo4jGraphStore",
        lambda config: Neo4jGraphStore(config, driver=driver),
    )

    client = TestClient(create_app(_build_config()))
    response = client.get("/api/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["label"] == "Haystack"
    assert body["edges"][0]["relation"] == "VERWENDET"


def test_upload_rejects_non_pdf() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post(
        "/api/upload",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_empty_file() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post(
        "/api/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_index_serves_html() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Graph RAG" in response.text
    assert "drop-area" in response.text
    assert "chat-input" in response.text
    assert 'data-view="chat"' in response.text


class FakeQueryPipeline:
    last_call: dict | None = None

    def __init__(self, config) -> None:
        self.config = config
        self.store = type("S", (), {"close": lambda self: None})()

    def run(self, question, *, top_k=None, hops=None):
        FakeQueryPipeline.last_call = {
            "question": question,
            "top_k": top_k,
            "hops": hops,
        }
        return QueryResult(
            answer=f"Antwort auf: {question}",
            context="merged-context",
            vector_documents=[make_document("v1", meta={"chunk_id": "c1"})],
            graph_documents=[
                make_document("g1", meta={"chunk_id": "c2"}),
                make_document("g2", meta={"chunk_id": "c3"}),
            ],
            entity_context="A --rel--> B",
            query_entities=["Haystack", "Neo4j"],
        )


def test_query_endpoint_returns_answer(monkeypatch) -> None:
    FakeQueryPipeline.last_call = None
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))

    response = client.post("/api/query", json={"question": "Was ist Haystack?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Antwort auf: Was ist Haystack?"
    assert body["query_entities"] == ["Haystack", "Neo4j"]
    assert body["vector_chunks"] == 1
    assert body["graph_chunks"] == 2
    assert FakeQueryPipeline.last_call == {
        "question": "Was ist Haystack?",
        "top_k": None,
        "hops": None,
    }


def test_query_endpoint_forwards_overrides(monkeypatch) -> None:
    FakeQueryPipeline.last_call = None
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))

    response = client.post(
        "/api/query",
        json={"question": "test", "top_k": 5, "hops": 2},
    )

    assert response.status_code == 200
    assert FakeQueryPipeline.last_call == {"question": "test", "top_k": 5, "hops": 2}


def test_query_endpoint_rejects_empty_question() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code == 422


def test_query_endpoint_rejects_blank_question(monkeypatch) -> None:
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 400
    assert "leer" in response.json()["detail"]


def test_query_endpoint_rejects_invalid_hops() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": "x", "hops": 9})
    assert response.status_code == 422

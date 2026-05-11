from __future__ import annotations

from fastapi.testclient import TestClient

from kg_rag.config import HuggingFaceConfig, Neo4jConfig, RagConfig
from kg_rag.neo4j_store import Neo4jGraphStore
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

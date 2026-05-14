from __future__ import annotations

import time

from fastapi.testclient import TestClient

from kg_rag.compat import make_document
from kg_rag.config import LLMConfig, Neo4jConfig, RagConfig
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines.query import QueryResult
from kg_rag.web import app as web_app
from kg_rag.web.app import INDEXING_JOBS, JOBS_LOCK, create_app


TEST_SESSION_ID = "test-session-12345"
SESSION_HEADERS = {"X-Session-Id": TEST_SESSION_ID}


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
        llm=LLMConfig(api_key="sk-test"),
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
    response = client.get("/api/graph", headers=SESSION_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["label"] == "Haystack"
    assert body["edges"][0]["relation"] == "VERWENDET"


def test_graph_endpoint_rejects_missing_session() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.get("/api/graph")
    assert response.status_code == 400
    assert "Session" in response.json()["detail"]


def test_upload_rejects_non_pdf() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post(
        "/api/upload",
        files={"file": ("note.txt", b"hello", "text/plain")},
        headers=SESSION_HEADERS,
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_empty_file() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post(
        "/api/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers=SESSION_HEADERS,
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

    def run(self, question, *, top_k=None, hops=None, session_id=None):
        FakeQueryPipeline.last_call = {
            "question": question,
            "top_k": top_k,
            "hops": hops,
            "session_id": session_id,
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
            citations=[],
        )


def test_query_endpoint_returns_answer(monkeypatch) -> None:
    FakeQueryPipeline.last_call = None
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))

    response = client.post(
        "/api/query",
        json={"question": "Was ist Haystack?"},
        headers=SESSION_HEADERS,
    )

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
        "session_id": TEST_SESSION_ID,
    }


def test_query_endpoint_forwards_overrides(monkeypatch) -> None:
    FakeQueryPipeline.last_call = None
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))

    response = client.post(
        "/api/query",
        json={"question": "test", "top_k": 5, "hops": 2},
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 200
    assert FakeQueryPipeline.last_call == {
        "question": "test",
        "top_k": 5,
        "hops": 2,
        "session_id": TEST_SESSION_ID,
    }


def test_query_endpoint_rejects_empty_question() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": ""}, headers=SESSION_HEADERS)
    assert response.status_code == 422


def test_query_endpoint_rejects_blank_question(monkeypatch) -> None:
    monkeypatch.setattr("kg_rag.web.app.QueryPipeline", FakeQueryPipeline)
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": "   "}, headers=SESSION_HEADERS)
    assert response.status_code == 400
    assert "leer" in response.json()["detail"]


def test_query_endpoint_rejects_invalid_hops() -> None:
    client = TestClient(create_app(_build_config()))
    response = client.post("/api/query", json={"question": "x", "hops": 9}, headers=SESSION_HEADERS)
    assert response.status_code == 422


def _clear_jobs() -> None:
    with JOBS_LOCK:
        INDEXING_JOBS.clear()


def test_upload_returns_job_id_and_202(monkeypatch) -> None:
    _clear_jobs()
    submitted: list = []

    class FakeFuture:
        def __init__(self) -> None:
            self.cancelled_flag = False

    def fake_submit(fn, *args, **kwargs):
        submitted.append((fn, args, kwargs))
        return FakeFuture()

    monkeypatch.setattr(web_app.JOB_EXECUTOR, "submit", fake_submit)

    client = TestClient(create_app(_build_config()))
    response = client.post(
        "/api/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["filename"] == "doc.pdf"
    assert isinstance(body["job_id"], str) and len(body["job_id"]) > 16
    assert body["estimated_seconds"] >= 30
    assert len(submitted) == 1

    with JOBS_LOCK:
        state = INDEXING_JOBS[body["job_id"]]
    assert state.session_id == TEST_SESSION_ID
    assert state.status == "queued"
    assert state.tmp_dir is not None and state.tmp_dir.exists()


def test_jobs_endpoint_returns_state(monkeypatch) -> None:
    _clear_jobs()
    monkeypatch.setattr(web_app.JOB_EXECUTOR, "submit", lambda *a, **k: None)

    client = TestClient(create_app(_build_config()))
    upload_response = client.post(
        "/api/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
        headers=SESSION_HEADERS,
    )
    job_id = upload_response.json()["job_id"]

    response = client.get(f"/api/jobs/{job_id}", headers=SESSION_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["filename"] == "doc.pdf"
    assert body["graph"] is None


def test_jobs_endpoint_returns_404_for_unknown_id() -> None:
    _clear_jobs()
    client = TestClient(create_app(_build_config()))
    response = client.get("/api/jobs/00000000000000000000000000000000", headers=SESSION_HEADERS)
    assert response.status_code == 404


def test_jobs_endpoint_isolates_by_session(monkeypatch) -> None:
    _clear_jobs()
    monkeypatch.setattr(web_app.JOB_EXECUTOR, "submit", lambda *a, **k: None)

    client = TestClient(create_app(_build_config()))
    upload_response = client.post(
        "/api/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
        headers=SESSION_HEADERS,
    )
    job_id = upload_response.json()["job_id"]

    foreign_headers = {"X-Session-Id": "other-session-99999"}
    response = client.get(f"/api/jobs/{job_id}", headers=foreign_headers)
    assert response.status_code == 404


def test_jobs_endpoint_requires_session() -> None:
    _clear_jobs()
    client = TestClient(create_app(_build_config()))
    response = client.get("/api/jobs/anything")
    assert response.status_code == 400


def test_run_indexing_job_marks_done_on_success(monkeypatch, tmp_path) -> None:
    _clear_jobs()

    class FakeStore:
        def close(self) -> None:
            pass

    class FakePipeline:
        def __init__(self, config) -> None:
            self.store = FakeStore()

        def run(self, paths, *, session_id, progress):
            progress("parsing", 0, 0)
            progress("extracting", 0, 2)
            progress("extracting", 2, 2)
            progress("persisting", 2, 2)
            progress("done", 2, 2)
            return 2

    monkeypatch.setattr(web_app, "IndexingPipeline", FakePipeline)
    monkeypatch.setattr(web_app, "_fetch_graph_unsafe", lambda config, sid: {"nodes": [], "edges": []})

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    job_id = "abc1234567890def1234567890123456"
    from kg_rag.web.app import JobState

    with JOBS_LOCK:
        INDEXING_JOBS[job_id] = JobState(
            id=job_id,
            session_id=TEST_SESSION_ID,
            filename="doc.pdf",
            status="queued",
            tmp_dir=tmp_path,
        )

    web_app._run_indexing_job(job_id, pdf_path, _build_config())

    with JOBS_LOCK:
        state = INDEXING_JOBS[job_id]
    assert state.status == "done"
    assert state.chunks_indexed == 2
    assert state.graph == {"nodes": [], "edges": []}
    assert state.finished_at is not None


def test_run_indexing_job_marks_error_on_failure(monkeypatch, tmp_path) -> None:
    _clear_jobs()

    class BrokenPipeline:
        def __init__(self, config) -> None:
            self.store = type("S", (), {"close": lambda self: None})()

        def run(self, paths, *, session_id, progress):
            raise RuntimeError("boom")

    monkeypatch.setattr(web_app, "IndexingPipeline", BrokenPipeline)

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    job_id = "def1234567890abc1234567890123456"
    from kg_rag.web.app import JobState

    with JOBS_LOCK:
        INDEXING_JOBS[job_id] = JobState(
            id=job_id,
            session_id=TEST_SESSION_ID,
            filename="doc.pdf",
            status="queued",
            tmp_dir=tmp_path,
        )

    web_app._run_indexing_job(job_id, pdf_path, _build_config())

    with JOBS_LOCK:
        state = INDEXING_JOBS[job_id]
    assert state.status == "error"
    assert state.error == "boom"
    assert state.finished_at is not None


def test_session_end_deletes_session(monkeypatch) -> None:
    deleted_ids: list[str] = []

    class TrackingDriver:
        def session(self, database):
            class S:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *_args):
                    return None

                def run(self_inner, query, **params):
                    if "DETACH DELETE" in query and "session_id" in params:
                        deleted_ids.append(params["session_id"])
                    return FakeResult([])

            return S()

        def close(self):
            pass

    monkeypatch.setattr(
        "kg_rag.web.app.Neo4jGraphStore",
        lambda config: Neo4jGraphStore(config, driver=TrackingDriver()),
    )

    client = TestClient(create_app(_build_config()))
    response = client.post("/api/session/end", json={"session_id": TEST_SESSION_ID})

    assert response.status_code == 200
    assert response.json() == {"deleted": TEST_SESSION_ID}
    assert deleted_ids == [TEST_SESSION_ID]


def test_document_text_endpoint_returns_pages(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "kg_rag.web.app._resolve_pdf_file",
        lambda session_id, document_id: pdf_path,
    )
    monkeypatch.setattr(
        "kg_rag.web.app._extract_pdf_pages",
        lambda path: [{"page_number": 1, "text": "Seite eins"}, {"page_number": 2, "text": "Seite zwei"}],
    )

    client = TestClient(create_app(_build_config()))
    response = client.get(
        "/api/document/abcdef1234567890abcdef1234567890/text",
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["pages"]) == 2
    assert body["pages"][0]["page_number"] == 1
    assert body["pages"][1]["text"] == "Seite zwei"


def test_document_text_endpoint_returns_404_for_unknown(monkeypatch) -> None:
    monkeypatch.setattr("kg_rag.web.app._resolve_pdf_file", lambda *_: None)

    client = TestClient(create_app(_build_config()))
    response = client.get(
        "/api/document/abcdef1234567890abcdef1234567890/text",
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 404

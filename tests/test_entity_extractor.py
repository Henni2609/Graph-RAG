import threading
import time

from kg_rag.compat import make_document
from kg_rag.components.entity_extractor import EntityExtractor, parse_extraction_response


def test_parse_extraction_response_from_fenced_json() -> None:
    raw = """```json
    {
      "entities": [
        {"name": "Haystack", "type": "Technologie", "description": "RAG framework"},
        {"name": "Neo4j", "type": "Technologie", "description": "Graph database"},
        {"name": "Haystack", "type": "Technologie", "description": "duplicate"}
      ],
      "relations": [
        {"source": "Haystack", "target": "Neo4j", "relation": "VERWENDET"},
        {"source": "Unknown", "target": "Neo4j", "relation": "VERWENDET"}
      ]
    }
    ```"""

    result = parse_extraction_response(raw)

    assert [entity.name for entity in result.entities] == ["Haystack", "Neo4j"]
    assert len(result.relations) == 1
    assert result.relations[0].source == "Haystack"
    assert result.relations[0].target == "Neo4j"


def test_parse_extraction_response_returns_empty_result_for_invalid_json() -> None:
    result = parse_extraction_response("No JSON here")

    assert result.entities == []
    assert result.relations == []


class _CountingGenerator:
    """Fake generator with bounded concurrency observation."""

    def __init__(self, sleep_seconds: float = 0.05) -> None:
        self.sleep_seconds = sleep_seconds
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self._lock = threading.Lock()

    def run(self, *, messages, generation_kwargs=None):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.sleep_seconds)
            user_text = messages[-1].text if hasattr(messages[-1], "text") else str(messages[-1])
            chunk_id = user_text.split("chunk-", 1)[-1].split("\n", 1)[0].strip()
            return {
                "replies": [
                    f'{{"entities":[{{"name":"E{chunk_id}","type":"Konzept","description":""}}],"relations":[]}}'
                ],
            }
        finally:
            with self._lock:
                self.active -= 1


def test_entity_extractor_preserves_order_and_runs_concurrently() -> None:
    documents = [make_document(f"chunk-{i}", meta={"chunk_id": f"c{i}"}) for i in range(8)]
    generator = _CountingGenerator(sleep_seconds=0.08)
    extractor = EntityExtractor(generator=generator, concurrency=4)

    result = extractor.run(documents)["documents"]

    assert len(result) == 8
    for i, doc in enumerate(result):
        assert doc.meta["entities"][0]["name"] == f"E{i}"
        assert doc.meta["chunk_id"] == f"c{i}"
    assert generator.calls == 8
    assert generator.peak_active >= 2, "extractor did not run calls in parallel"
    assert generator.peak_active <= 4, "extractor exceeded the concurrency limit"


def test_entity_extractor_isolates_per_chunk_failures() -> None:
    documents = [
        make_document("chunk-0", meta={"chunk_id": "c0"}),
        make_document("chunk-1", meta={"chunk_id": "c1"}),
        make_document("chunk-2", meta={"chunk_id": "c2"}),
    ]

    class FlakyGenerator:
        def run(self, *, messages, generation_kwargs=None):
            user_text = messages[-1].text if hasattr(messages[-1], "text") else str(messages[-1])
            if "chunk-1" in user_text:
                raise RuntimeError("boom")
            return {"replies": ['{"entities":[{"name":"X","type":"Konzept","description":""}],"relations":[]}']}

    extractor = EntityExtractor(generator=FlakyGenerator(), concurrency=2)
    result = extractor.run(documents)["documents"]

    assert [doc.meta["entities"] for doc in result] == [
        [{"name": "X", "type": "Konzept", "description": ""}],
        [],
        [{"name": "X", "type": "Konzept", "description": ""}],
    ]


def test_entity_extractor_handles_empty_input() -> None:
    extractor = EntityExtractor(generator=_CountingGenerator(), concurrency=4)
    assert extractor.run([]) == {"documents": []}


def test_entity_extractor_emits_progress_events() -> None:
    documents = [make_document(f"chunk-{i}", meta={"chunk_id": f"c{i}"}) for i in range(6)]
    generator = _CountingGenerator(sleep_seconds=0.02)
    extractor = EntityExtractor(generator=generator, concurrency=3)

    events: list[tuple[str, int, int]] = []
    extractor.run(documents, progress=lambda step, c, t: events.append((step, c, t)))

    assert events[0] == ("extracting", 0, 6)
    assert events[-1] == ("extracting", 6, 6)
    counters = [c for _step, c, _t in events]
    assert counters == sorted(counters), "progress counter must be monotonically non-decreasing"
    assert all(step == "extracting" and total == 6 for step, _c, total in events)


def test_entity_extractor_empty_input_still_fires_zero_event() -> None:
    events: list[tuple[str, int, int]] = []
    EntityExtractor(generator=_CountingGenerator(), concurrency=2).run(
        [],
        progress=lambda step, c, t: events.append((step, c, t)),
    )
    assert events == [("extracting", 0, 0)]

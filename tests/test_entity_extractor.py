from kg_rag.components.entity_extractor import parse_extraction_response


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

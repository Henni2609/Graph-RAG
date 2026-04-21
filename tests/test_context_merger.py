from kg_rag.compat import make_document
from kg_rag.components.context_merger import ContextMerger


def test_context_merger_deduplicates_with_vector_priority() -> None:
    vector_doc = make_document(
        "Vector text",
        meta={"chunk_id": "chunk-1", "source": "a.md", "chunk_index": 0},
    )
    duplicate_graph_doc = make_document(
        "Graph duplicate text",
        meta={"chunk_id": "chunk-1", "source": "a.md", "chunk_index": 0},
    )
    graph_doc = make_document(
        "Graph text",
        meta={"chunk_id": "chunk-2", "source": "b.md", "chunk_index": 1},
    )

    result = ContextMerger(max_context_chars=1000).run(
        vector_docs=[vector_doc],
        graph_docs=[duplicate_graph_doc, graph_doc],
        entity_context="- Haystack --VERWENDET--> Neo4j",
    )

    assert len(result["documents"]) == 2
    assert "Vector text" in result["merged_context"]
    assert "Graph duplicate text" not in result["merged_context"]
    assert "[Semantisch relevant]" in result["merged_context"]
    assert "[Via Graph-Traversal]" in result["merged_context"]
    assert "[Entity-Relationen]" in result["merged_context"]


def test_context_merger_respects_character_limit() -> None:
    doc = make_document(
        "A" * 200,
        meta={"chunk_id": "chunk-1", "source": "a.md", "chunk_index": 0},
    )

    context = ContextMerger(max_context_chars=80).run([doc], [], "")["merged_context"]

    assert len(context) <= 80

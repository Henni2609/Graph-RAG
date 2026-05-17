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


def test_context_merger_prioritises_high_relevance_middle_chunk() -> None:
    """A high-cosine vector chunk from the document middle must survive budget trimming."""
    intro = make_document(
        "I" * 100,
        meta={"chunk_id": "c0", "source": "doc.pdf", "chunk_index": 0},
        score=0.3,
    )
    relevant = make_document(
        "R" * 100,
        meta={"chunk_id": "c5", "source": "doc.pdf", "chunk_index": 5},
        score=0.9,
    )
    conclusion = make_document(
        "C" * 100,
        meta={"chunk_id": "c9", "source": "doc.pdf", "chunk_index": 9},
        score=0.4,
    )
    # Each section header is ~60 chars; total ~3 * (60+100+2) = ~486. Budget = 330 fits 2.
    result = ContextMerger(max_context_chars=330).run(
        vector_docs=[intro, relevant, conclusion],
        graph_docs=[],
        entity_context="",
    )
    ctx = result["merged_context"]
    assert "R" * 100 in ctx, "High-relevance middle chunk must be in context"
    citations = result["citations"]
    # Citations must be consecutively numbered 1..k with no gaps.
    assert [c["index"] for c in citations] == list(range(1, len(citations) + 1))

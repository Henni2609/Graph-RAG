from kg_rag.compat import document_meta, make_document
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


def test_context_merger_renders_in_document_order() -> None:
    """Chunks from the same document must appear in chunk_index order (document coherence).
    The most-relevant document still surfaces first when multiple documents are present."""
    low = make_document(
        "Low relevance text",
        meta={"chunk_id": "c0", "source": "doc.pdf", "chunk_index": 0, "document_id": "doc1"},
        score=0.4,
    )
    high = make_document(
        "High relevance text",
        meta={"chunk_id": "c5", "source": "doc.pdf", "chunk_index": 5, "document_id": "doc1"},
        score=0.9,
    )
    mid = make_document(
        "Mid relevance text",
        meta={"chunk_id": "c3", "source": "doc.pdf", "chunk_index": 3, "document_id": "doc1"},
        score=0.6,
    )
    result = ContextMerger(max_context_chars=10000).run(
        vector_docs=[low, high, mid],
        graph_docs=[],
        entity_context="",
    )
    ctx = result["merged_context"]
    # Within a single document, chunks must appear in chunk_index order (0, 3, 5).
    assert ctx.index("Low relevance text") < ctx.index("Mid relevance text")
    assert ctx.index("Mid relevance text") < ctx.index("High relevance text")
    citations = result["citations"]
    assert [c["index"] for c in citations] == list(range(1, len(citations) + 1))


def test_filter_by_similarity_excludes_low_score_chunks() -> None:
    """Chunks below threshold are dropped; chunks with no score pass through."""
    from kg_rag.pipelines.query import _filter_by_similarity

    high = make_document("kept", meta={"chunk_id": "c1"}, score=0.8)
    low = make_document("dropped", meta={"chunk_id": "c2"}, score=0.1)
    no_score = make_document("pass-through", meta={"chunk_id": "c3"})

    result = _filter_by_similarity([high, low, no_score], threshold=0.25)
    ids = {document_meta(d).get("chunk_id") for d in result}
    assert "c1" in ids
    assert "c2" not in ids
    assert "c3" in ids

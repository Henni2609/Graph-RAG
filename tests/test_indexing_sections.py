from __future__ import annotations

from kg_rag.pipelines.indexing import _segment_into_sections, _split_into_pages, normalize_chunk_metadata
from kg_rag.compat import make_document, document_content


def test_segment_detects_roman_headings():
    page = (
        "Some preamble text here.\n"
        "I. Introduction\n"
        "This is the intro body.\n"
        "VII. Conclusion\n"
        "Prompt engineering is game-changing."
    )
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments]
    assert "I. Introduction" in titles
    assert "VII. Conclusion" in titles


def test_segment_preamble_has_no_title():
    page = "Some preamble text.\nI. Introduction\nBody."
    segments = _segment_into_sections(page)
    assert segments[0][0] is None
    assert "Some preamble text." in segments[0][1]


def test_segment_no_headings_returns_single_segment():
    page = "Just a plain paragraph. No headings here at all."
    segments = _segment_into_sections(page)
    assert len(segments) == 1
    assert segments[0][0] is None


def test_sentence_ending_in_period_not_a_heading():
    page = "Prompt engineering improves AI outputs significantly.\nVII. Conclusion\nBody."
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert not any("Prompt engineering" in (t or "") for t in titles)
    assert "VII. Conclusion" in titles


def test_segment_detects_keyword_headings():
    page = "Abstract\nThis paper covers prompt engineering.\nReferences\n[1] Smith et al."
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert "Abstract" in titles
    assert "References" in titles


# ── German keyword headings ────────────────────────────────────────────────────

def test_segment_detects_german_keyword_headings():
    page = (
        "Zusammenfassung\nDiese Arbeit untersucht Prompt Engineering.\n"
        "Einleitung\nKI-Systeme sind komplex.\n"
        "Fazit\nPrompt Engineering ist entscheidend."
    )
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert "Zusammenfassung" in titles
    assert "Einleitung" in titles
    assert "Fazit" in titles


def test_segment_detects_german_methods_and_results():
    page = "Methodik\nWir verwendeten Neo4j.\nErgebnisse\nDie Präzision stieg."
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert any("Methodik" in (t or "") for t in titles)
    assert any("Ergebnisse" in (t or "") for t in titles)


def test_segment_detects_literatur():
    page = "Literaturverzeichnis\n[1] Braun et al."
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert any("Literatur" in (t or "") for t in titles)


def test_segment_german_schlussfolgerung():
    page = "Schlussfolgerungen\nDie Methode ist effektiv."
    segments = _segment_into_sections(page)
    titles = [t for t, _ in segments if t]
    assert any("Schlussfolgerung" in (t or "") for t in titles)


# ── normalize_chunk_metadata ───────────────────────────────────────────────────

def test_normalize_prefixes_section_title_into_first_chunk_only():
    """The first chunk of each section gets the heading prepended so that
    heading-targeted queries (e.g. German 'Schlussfolgerung' → English 'Conclusion')
    have a cross-lingual anchor. Subsequent chunks in the same section are left
    unchanged to avoid skewing every chunk with the same heading vector."""
    first = make_document("Prompt engineering is game-changing.", meta={
        "source": "/tmp/test.pdf",
        "page_number": 5,
        "section_title": "VII. Conclusion",
    })
    second = make_document("Organizations can leverage AI through strategic prompting.", meta={
        "source": "/tmp/test.pdf",
        "page_number": 5,
        "section_title": "VII. Conclusion",
    })
    result = normalize_chunk_metadata([first, second], session_id="test")
    assert len(result) == 2
    first_content = result[0].content if hasattr(result[0], "content") else result[0].page_content
    second_content = result[1].content if hasattr(result[1], "content") else result[1].page_content
    # First chunk: section title prepended as cross-lingual anchor
    assert first_content.startswith("VII. Conclusion\n\n")
    assert "Prompt engineering is game-changing." in first_content
    # Second chunk: raw content unchanged — no heading vector skew
    assert not second_content.startswith("VII. Conclusion")
    assert "Organizations can leverage AI" in second_content
    # section_title still accessible via meta on both chunks
    assert result[0].meta["section_title"] == "VII. Conclusion"
    assert result[1].meta["section_title"] == "VII. Conclusion"


def test_normalize_no_section_title_unchanged():
    doc = make_document("Just a normal chunk.", meta={"source": "/tmp/test.pdf"})
    result = normalize_chunk_metadata([doc], session_id="test")
    content = result[0].content if hasattr(result[0], "content") else result[0].page_content
    assert not content.startswith("\n")
    assert "Just a normal chunk." in content


def test_normalize_chunk_id_is_position_stable():
    """chunk_id must depend only on session|document|index, not content,
    so re-indexing the same position merges rather than orphans the old chunk."""
    doc = make_document("Some content.", meta={"source": "/tmp/stable.pdf"})
    result1 = normalize_chunk_metadata([doc], session_id="s1")
    # Same position, different content
    doc2 = make_document("Changed content.", meta={"source": "/tmp/stable.pdf"})
    result2 = normalize_chunk_metadata([doc2], session_id="s1")
    assert result1[0].meta["chunk_id"] == result2[0].meta["chunk_id"]


# ── ToC page detection ─────────────────────────────────────────────────────────

_TOC_PAGE = (
    "5 Post-Training 28\n"
    "5.1 Post-Training Pipeline . . . . . . . . 28\n"
    "5.2 Post-Training Infrastructures . . . . 33\n"
    "5.3 Standard Benchmark Evaluation . . . . 36\n"
    "5.4 Performance on Real-World Tasks . . . 41\n"
    "5.4.4 Code Agent . . . . . . . . . . . . 44\n"
    "6 Conclusion, Limitations, and Future Directions 44\n"
    "A Author List and Acknowledgment 54\n"
    "A.1 Author List . . . . . . . . . . . . 54\n"
    "A.2 Acknowledgment . . . . . . . . . . 55\n"
    "B Evaluation Details 55\n"
)


def test_toc_page_produces_single_untitled_segment():
    segments = _segment_into_sections(_TOC_PAGE)
    assert len(segments) == 1
    title, body = segments[0]
    assert title is None
    assert "Conclusion" in body


def test_toc_page_segment_title_never_contains_section_names():
    segments = _segment_into_sections(_TOC_PAGE)
    titles = [t for t, _ in segments if t]
    assert not any("Conclusion" in (t or "") for t in titles)
    assert not any("Post-Training" in (t or "") for t in titles)


def test_real_conclusion_page_still_segmented():
    page = (
        "6 Conclusion, Limitations, and Future Directions\n"
        "We have presented DeepSeek-V4, a strong Mixture-of-Experts language model "
        "comprising 671B total parameters with 37B activated for each token. "
        "The pre-training of DeepSeek-V4 requires only 2.788M H800 GPU hours. "
        "Despite the economical training costs, the comprehensive evaluation "
        "reveals that DeepSeek-V4 achieves state-of-the-art performance among "
        "open-source models and is competitive with leading closed-source models.\n"
    )
    segments = _segment_into_sections(page)
    titled = [(t, b) for t, b in segments if t]
    assert titled, "expected at least one titled segment on a real content page"
    assert any("Conclusion" in (t or "") for t, _ in titled)
    # Body must contain actual prose, not be empty
    for _, body in titled:
        assert len(body) > 50


def test_empty_body_heading_does_not_clone_page():
    # Page whose only heading has no following text — should produce no document
    # for that heading (the empty-body segment is skipped in _split_into_pages).
    page = "Some real content here.\n3 Methods\n"
    docs = _split_into_pages(page, "/tmp/test.pdf", "sess")
    # The "3 Methods" heading has no body → it must not clone the whole page
    full_page_clones = [
        d for d in docs
        if document_content(d).strip() == page.strip() and d.meta.get("section_title") == "3 Methods"
    ]
    assert not full_page_clones

from __future__ import annotations

from kg_rag.pipelines.indexing import _segment_into_sections, normalize_chunk_metadata
from kg_rag.compat import make_document


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

def test_normalize_does_not_prefix_section_title_into_content():
    """section_title must be stored in meta only — not prepended to the embedded text.
    Prepending skews every chunk of a long section with the same heading vector."""
    doc = make_document("Prompt engineering is game-changing.", meta={
        "source": "/tmp/test.pdf",
        "page_number": 5,
        "section_title": "VII. Conclusion",
    })
    result = normalize_chunk_metadata([doc], session_id="test")
    assert len(result) == 1
    content = result[0].content if hasattr(result[0], "content") else result[0].page_content
    # Raw content must be preserved unchanged
    assert "Prompt engineering is game-changing." in content
    # Section title must NOT be prepended into the embedded text
    assert not content.startswith("VII. Conclusion\n")
    # But it must remain accessible via meta for the context header display
    assert result[0].meta["section_title"] == "VII. Conclusion"


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

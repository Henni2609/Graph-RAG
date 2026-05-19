import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from kg_rag.compat import make_document
from kg_rag.pipelines.indexing import (
    _ocr_pages,
    collect_supported_files,
    fallback_sentence_split,
    normalize_chunk_metadata,
)


def test_collect_supported_files_recurses_and_filters(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("C", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_text("fake", encoding="utf-8")

    files = collect_supported_files([tmp_path])

    assert [path.name for path in files] == ["a.md", "b.txt", "c.pdf"]


def test_fallback_sentence_split_uses_overlap() -> None:
    document = make_document(
        "One. Two. Three. Four.",
        meta={"source": "/tmp/doc.md", "title": "doc.md"},
    )

    chunks = fallback_sentence_split([document], split_length=2, split_overlap=1)

    assert [chunk.content for chunk in chunks] == ["One. Two.", "Two. Three.", "Three. Four.", "Four."]


def test_normalize_chunk_metadata_adds_stable_ids() -> None:
    chunk = make_document("Content", meta={"source": "/tmp/doc.md"})

    normalized = normalize_chunk_metadata([chunk])[0]

    assert normalized.meta["document_id"]
    assert normalized.meta["chunk_id"]
    assert normalized.meta["chunk_index"] == 0
    assert normalized.meta["title"] == "doc.md"


def test_ocr_pages_skips_covered_pages(monkeypatch, tmp_path: Path) -> None:
    mock_tes = MagicMock()
    monkeypatch.setitem(sys.modules, "pytesseract", mock_tes)

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.0")

    mock_page = MagicMock()
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        docs = _ocr_pages(pdf_path, "test", covered_pages={1}, ocr_language="eng")

    assert docs == []
    mock_tes.image_to_string.assert_not_called()


def test_ocr_pages_produces_document_for_uncovered_image_page(monkeypatch, tmp_path: Path) -> None:
    mock_tes = MagicMock()
    mock_tes.image_to_string.return_value = "Scanned text"
    mock_tes.TesseractNotFoundError = Exception
    monkeypatch.setitem(sys.modules, "pytesseract", mock_tes)

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.0")

    mock_img = MagicMock()
    mock_img.image = MagicMock()
    mock_page = MagicMock()
    mock_page.images = [mock_img]
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        docs = _ocr_pages(pdf_path, "test", covered_pages=set(), ocr_language="eng")

    assert len(docs) == 1
    assert docs[0].content == "Scanned text"
    assert docs[0].meta["page_number"] == 1
    assert docs[0].meta["extraction"] == "ocr"


def test_ocr_pages_graceful_when_pytesseract_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.0")

    docs = _ocr_pages(pdf_path, "test", covered_pages=set(), ocr_language="eng")

    assert docs == []

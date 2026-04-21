from pathlib import Path

from kg_rag.compat import make_document
from kg_rag.pipelines.indexing import collect_supported_files, fallback_sentence_split, normalize_chunk_metadata


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
    chunk = make_document("Content", meta={"source": "/tmp/doc.md", "split_idx": 3})

    normalized = normalize_chunk_metadata([chunk])[0]

    assert normalized.meta["document_id"]
    assert normalized.meta["chunk_id"]
    assert normalized.meta["chunk_index"] == 3
    assert normalized.meta["title"] == "doc.md"

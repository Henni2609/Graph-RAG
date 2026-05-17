from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

# Matches academic section headings (short lines, no sentence-ending punctuation).
# Groups: roman-numeral numbered  "VII. Conclusion",
#         decimal numbered        "5.2 Managing AI Biases",
#         bare keyword headings   English + German equivalents
_HEADING_RE = re.compile(
    r"^(?:"
    r"[IVXLC]{1,6}\.\s+\S"           # Roman numeral: "I. Introduction"
    r"|(?:\d+)(?:\.\d+)*\.?\s+\S"    # Decimal:       "5.2 Managing …" / "5."
    r"|(?:"
    # English keywords
    r"Abstract|Keywords?|References?|Conclusions?|Introduction|Acknowledg(?:e)?ments?"
    r"|"
    # German keywords
    r"Zusammenfassung|Einleitung|Schlussfolgerung(?:en)?|Fazit|"
    r"Literatur(?:verzeichnis)?|Danksagung|Methodik|Methoden?|Ergebnisse?|"
    r"Diskussion|Materialien?|Anhang|Abk(?:ü|ue)rzungen?"
    r")"
    r").*$",
    re.IGNORECASE,
)

from kg_rag.compat import Document, document_content, document_meta, make_document
from kg_rag.components.entity_extractor import EntityExtractor
from kg_rag.config import RagConfig
from kg_rag.logging import logger
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore, stable_id


ProgressCallback = Callable[[str, int, int], None]


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class IndexingPipeline:
    def __init__(
        self,
        config: RagConfig,
        *,
        store: Neo4jGraphStore | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.config = config
        self.store = store or Neo4jGraphStore(config.neo4j)
        self.entity_extractor = entity_extractor or EntityExtractor(
            llm_config=config.llm,
            max_tokens=config.entity_max_tokens,
            concurrency=config.extraction_concurrency,
        )

    def run(
        self,
        paths: Iterable[str | Path],
        *,
        overwrite: bool = False,
        session_id: str = DEFAULT_SESSION_ID,
        progress: ProgressCallback | None = None,
    ) -> int:
        def emit(step: str, current: int = 0, total: int = 0) -> None:
            if progress is not None:
                progress(step, current, total)

        files = collect_supported_files(paths)
        logger.info(f"Indexing {len(files)} file(s) for session {session_id}")
        if not files:
            emit("done", 0, 0)
            return 0

        emit("parsing")
        documents = load_documents(files, session_id=session_id)

        emit("splitting")
        chunks = split_documents(
            documents,
            split_length=self.config.chunk_split_length,
            split_overlap=self.config.chunk_split_overlap,
        )
        chunks = normalize_chunk_metadata(chunks, session_id=session_id)

        emit("embedding", 0, len(chunks))
        embedded_chunks = embed_documents(chunks, model=self.config.embedding_model)

        enriched_chunks = self.entity_extractor.run(embedded_chunks, progress=progress)["documents"]

        self.store.setup_schema(dimensions=self.config.embedding_dimensions)
        self.store.persist_documents(
            enriched_chunks,
            overwrite=overwrite,
            session_id=session_id,
            progress=progress,
        )

        # Per-document chunk reconciliation: remove stale chunks left from a
        # previous index run of the same document (e.g. after content edits).
        valid_chunks_by_doc: dict[str, set[str]] = defaultdict(set)
        for chunk in enriched_chunks:
            meta = document_meta(chunk)
            doc_id = meta.get("document_id", "")
            chunk_id = meta.get("chunk_id", "")
            if doc_id and chunk_id:
                valid_chunks_by_doc[doc_id].add(chunk_id)
        for doc_id, valid_ids in valid_chunks_by_doc.items():
            self.store.delete_stale_document_chunks(session_id, doc_id, valid_ids)

        # Persist embedding model metadata so the query pipeline can detect
        # model drift after the index has been built.
        self.store.store_indexing_meta(session_id, self.config.embedding_model, self.config.embedding_dimensions)

        emit("done", len(enriched_chunks), len(enriched_chunks))
        return len(enriched_chunks)


def collect_supported_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(dict.fromkeys(files))


def load_documents(files: list[Path], *, session_id: str = DEFAULT_SESSION_ID) -> list[Document]:
    text_files = [path for path in files if path.suffix.lower() in {".txt", ".md"}]
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
    documents: list[Document] = []
    documents.extend(_load_text_documents(text_files, session_id=session_id))
    documents.extend(_load_pdf_documents(pdf_files, session_id=session_id))
    return documents


def split_documents(
    documents: list[Document],
    *,
    split_length: int = 10,
    split_overlap: int = 2,
) -> list[Document]:
    if not documents:
        return []
    try:
        from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

        # remove_repeated_substrings=False: legitimate repeated content
        # (table headers, refrains) must not be silently stripped from the
        # text that ends up embedded and stored.
        cleaner = DocumentCleaner(remove_repeated_substrings=False)
        splitter = DocumentSplitter(
            split_by="sentence",
            split_length=split_length,
            split_overlap=split_overlap,
        )
        cleaned = cleaner.run(documents=documents)["documents"]
        return splitter.run(documents=cleaned)["documents"]
    except Exception as exc:
        logger.warning(f"Haystack splitter unavailable, using fallback sentence splitter: {exc}")
        return fallback_sentence_split(documents, split_length=split_length, split_overlap=split_overlap)


def embed_documents(documents: list[Document], *, model: str) -> list[Document]:
    if not documents:
        return []
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder

    embedder = SentenceTransformersDocumentEmbedder(model=model)
    embedder.warm_up()
    return embedder.run(documents=documents)["documents"]


def normalize_chunk_metadata(
    chunks: list[Document],
    *,
    session_id: str = DEFAULT_SESSION_ID,
) -> list[Document]:
    counters: dict[str, int] = defaultdict(int)
    normalized: list[Document] = []
    for chunk in chunks:
        meta = document_meta(chunk)
        source = str(meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown")
        document_id = str(meta.get("document_id") or stable_id(f"{session_id}|{source}"))
        chunk_index = int(meta.get("chunk_index", counters[document_id]) or 0)
        counters[document_id] = chunk_index + 1
        page_number = meta.get("page_number")
        try:
            page_number = int(page_number) if page_number is not None else 1
        except (TypeError, ValueError):
            page_number = 1
        section_title = meta.get("section_title") or ""
        raw_content = document_content(chunk)

        # Embed raw text only — section_title is stored in meta and shown in the
        # context header by ContextMerger. Prefixing it into the embedded text
        # skews every chunk in a long section with the same heading vector.
        content = raw_content

        # Use a position-stable chunk_id (session|doc|index) so that re-indexing
        # the same document at the same position updates the existing Chunk node
        # via MERGE rather than creating a parallel orphan node.
        chunk_id = meta.get("chunk_id") or stable_id(f"{session_id}|{document_id}|{chunk_index}")

        meta.update(
            {
                "source": source,
                "title": meta.get("title") or Path(source).name,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "page_number": max(1, page_number),
                "session_id": session_id,
                "chunk_id": chunk_id,
            }
        )
        normalized.append(make_document(content, meta=meta))
    return normalized


def fallback_sentence_split(
    documents: list[Document],
    *,
    split_length: int,
    split_overlap: int,
) -> list[Document]:
    # Split on sentence boundaries, avoiding abbreviations (z.B., Dr., Nr.)
    # and decimal numbers to reduce incorrect splits.
    _SENT_SPLIT = re.compile(
        r"(?<!"           # negative lookbehind:
        r"(?:[A-Z][a-z])" # abbreviation like "Dr", "Nr", "St"
        r")"
        r"(?<!\d)"        # decimal: "3.14"
        r"(?<=[.!?])\s+"  # actual sentence boundary
    )

    chunks: list[Document] = []
    step = max(1, split_length - split_overlap)
    for document in documents:
        sentences = [part.strip() for part in _SENT_SPLIT.split(document_content(document)) if part.strip()]
        if not sentences:
            continue
        for index, start in enumerate(range(0, len(sentences), step)):
            text = " ".join(sentences[start : start + split_length]).strip()
            if not text:
                continue
            meta = document_meta(document)
            meta["split_idx"] = index
            chunks.append(make_document(text, meta=meta))
    return chunks


def _clean_page_text(text: str) -> str:
    return re.sub(r"(?m)^\d{1,5}\s*$", "", text).strip()


def _segment_into_sections(page_text: str) -> list[tuple[str | None, str]]:
    """Split page_text at detected heading lines.

    Returns a list of (section_title, body) pairs. Text before the first
    heading has section_title=None. Headings must be ≤80 chars and must not
    end with sentence punctuation (.!?), to avoid false positives.
    """
    lines = page_text.splitlines()
    segments: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_heading = (
            stripped
            and len(stripped) <= 80
            and stripped[-1] not in ".!?"
            and _HEADING_RE.match(stripped) is not None
        )
        if is_heading:
            body = "\n".join(current_lines).strip()
            if body or current_title is not None:
                segments.append((current_title, body))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body or current_title is not None:
        segments.append((current_title, body))

    if not segments:
        return [(None, page_text)]
    return segments


def _split_into_pages(content: str, source: str, session_id: str) -> list[Document]:
    base_meta = _source_metadata(Path(source), session_id=session_id)
    documents: list[Document] = []
    for page_idx, page_text in enumerate(content.split("\x0c"), start=1):
        page_text = _clean_page_text(page_text)
        if not page_text:
            continue
        for section_title, body in _segment_into_sections(page_text):
            text = body if body else page_text
            meta = {**base_meta, "page_number": page_idx}
            if section_title:
                meta["section_title"] = section_title
            documents.append(make_document(text, meta=meta))
    return documents


def _load_text_documents(files: list[Path], *, session_id: str = DEFAULT_SESSION_ID) -> list[Document]:
    if not files:
        return []
    try:
        from haystack.components.converters.txt import TextFileToDocument

        converter = TextFileToDocument()
        result = converter.run(sources=files)
        documents = result["documents"]
        for index, document in enumerate(documents):
            source = _converted_source(document, files, index)
            _attach_source_metadata(document, source, session_id=session_id)
        return documents
    except Exception as exc:
        logger.warning(f"Haystack text converter unavailable, using fallback loader: {exc}")
        return [
            make_document(
                path.read_text(encoding="utf-8"),
                meta=_source_metadata(path, session_id=session_id),
            )
            for path in files
        ]


def _load_pdf_documents(files: list[Path], *, session_id: str = DEFAULT_SESSION_ID) -> list[Document]:
    if not files:
        return []
    try:
        from haystack.components.converters.pypdf import PyPDFToDocument

        converter = PyPDFToDocument()
        result = converter.run(sources=files)
        documents: list[Document] = []
        for index, raw_doc in enumerate(result["documents"]):
            source = str(files[index]) if index < len(files) else _converted_source(raw_doc, files, index)
            documents.extend(_split_into_pages(document_content(raw_doc), source, session_id))
        return documents
    except Exception as exc:
        logger.warning(f"Haystack PDF converter unavailable, using fallback loader: {exc}")
        from pypdf import PdfReader

        documents = []
        for path in files:
            reader = PdfReader(str(path))
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                documents.append(make_document(
                    text,
                    meta={**_source_metadata(path, session_id=session_id), "page_number": page_idx},
                ))
        return documents


def _attach_source_metadata(document: Document, source: str, *, session_id: str) -> None:
    meta = document_meta(document)
    meta.update(_source_metadata(Path(source), session_id=session_id))
    document.meta = meta


def _converted_source(document: Document, files: list[Path], index: int) -> str:
    if index < len(files):
        return str(files[index])
    meta = document_meta(document)
    for key in ("source", "file_path", "path"):
        if meta.get(key):
            return str(meta[key])
    return "unknown"


def _source_metadata(path: Path, *, session_id: str = DEFAULT_SESSION_ID) -> dict[str, str]:
    source = str(path.expanduser().resolve())
    return {
        "source": source,
        "title": path.name,
        "document_id": stable_id(f"{session_id}|{source}"),
        "session_id": session_id,
    }

from __future__ import annotations

import functools
import os
import re
from collections import defaultdict
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

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
from kg_rag.logging import log_timing, logger
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore, stable_id


ProgressCallback = Callable[[str, int, int], None]
FileCallback = Callable[[str, str], None]


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
            timeout=config.extraction_timeout_seconds,
            max_retries=config.extraction_max_retries,
        )

    def run(
        self,
        paths: Iterable[str | Path],
        *,
        overwrite: bool = False,
        session_id: str = DEFAULT_SESSION_ID,
        progress: ProgressCallback | None = None,
        on_file: FileCallback | None = None,
        on_pages_loaded: Callable[[list[Document]], None] | None = None,
    ) -> int:
        def emit(step: str, current: int = 0, total: int = 0) -> None:
            if progress is not None:
                progress(step, current, total)

        def emit_file(doc_id: str, status: str) -> None:
            if on_file is not None:
                on_file(doc_id, status)

        files = collect_supported_files(paths)
        logger.info(f"Indexing {len(files)} file(s) for session {session_id}")
        if not files:
            emit("done", 0, 0)
            return 0

        # Compute document_ids upfront — same formula as _source_metadata, which
        # uses already-resolved paths (collect_supported_files resolves them).
        file_doc_ids = [stable_id(f"{session_id}|{str(f)}") for f in files]

        for doc_id in file_doc_ids:
            emit_file(doc_id, "parsing")
        emit("parsing")
        with log_timing("parse"):
            documents = load_documents(
                files,
                session_id=session_id,
                ocr_enabled=self.config.ocr_enabled,
                ocr_language=self.config.ocr_language,
                ocr_workers=self.config.ocr_concurrency,
                ocr_min_text_chars=self.config.ocr_min_text_chars,
            )
        if on_pages_loaded is not None:
            on_pages_loaded(documents)
        for doc_id in file_doc_ids:
            emit_file(doc_id, "indexing")

        emit("splitting")
        with log_timing("split"):
            chunks = split_documents(
                documents,
                split_length=self.config.chunk_split_length,
                split_overlap=self.config.chunk_split_overlap,
            )
            chunks = normalize_chunk_metadata(chunks, session_id=session_id)

        emit("embedding", 0, len(chunks))
        with log_timing("embed"):
            embedded_chunks = embed_documents(
                chunks,
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
                device=self.config.embedding_device,
            )

        with log_timing("extract"):
            enriched_chunks = self.entity_extractor.run(embedded_chunks, progress=progress)["documents"]

        with log_timing("persist"):
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
        self.store.delete_stale_chunks_bulk(
            {doc_id: list(valid_ids) for doc_id, valid_ids in valid_chunks_by_doc.items()},
            session_id=session_id,
        )
        for doc_id in valid_chunks_by_doc:
            emit_file(doc_id, "done")

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


def load_documents(
    files: list[Path],
    *,
    session_id: str = DEFAULT_SESSION_ID,
    ocr_enabled: bool = True,
    ocr_language: str = "deu+eng",
    ocr_workers: int = 0,
    ocr_min_text_chars: int = 100,
) -> list[Document]:
    text_files = [path for path in files if path.suffix.lower() in {".txt", ".md"}]
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
    documents: list[Document] = []
    documents.extend(_load_text_documents(text_files, session_id=session_id))
    documents.extend(
        _load_pdf_documents(
            pdf_files,
            session_id=session_id,
            ocr_enabled=ocr_enabled,
            ocr_language=ocr_language,
            ocr_workers=ocr_workers,
            ocr_min_text_chars=ocr_min_text_chars,
        )
    )
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
        # Haystack's DocumentSplitter unconditionally overwrites page_number based on
        # form-feed counts within each parent document. Our parents already represent
        # a single page (no \f remaining), so each chunk would otherwise be tagged
        # page 1. Restore the parent's page_number via the splitter's source_id link.
        page_by_source = {d.id: document_meta(d).get("page_number") for d in cleaned}
        splits = splitter.run(documents=cleaned)["documents"]
        for split in splits:
            src_page = page_by_source.get(document_meta(split).get("source_id"))
            if src_page is not None:
                split.meta["page_number"] = src_page
        return splits
    except Exception as exc:
        logger.warning(f"Haystack splitter unavailable, using fallback sentence splitter: {exc}")
        return fallback_sentence_split(documents, split_length=split_length, split_overlap=split_overlap)


def _resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


@functools.lru_cache(maxsize=None)
def _get_doc_embedder(model: str, batch_size: int, device: str) -> Any:
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.utils import ComponentDevice

    embedder = SentenceTransformersDocumentEmbedder(
        model=model,
        batch_size=batch_size,
        device=ComponentDevice.from_str(device),
    )
    embedder.warm_up()
    return embedder


def embed_documents(
    documents: list[Document], *, model: str, batch_size: int = 64, device: str = "auto"
) -> list[Document]:
    if not documents:
        return []
    return _get_doc_embedder(model, batch_size, _resolve_device(device)).run(documents=documents)["documents"]


def normalize_chunk_metadata(
    chunks: list[Document],
    *,
    session_id: str = DEFAULT_SESSION_ID,
) -> list[Document]:
    counters: dict[str, int] = defaultdict(int)
    seen_sections: set[tuple[str, str]] = set()
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

        # Embed the section title into the first chunk of each section only —
        # gives heading-targeted queries a cross-lingual anchor without skewing
        # every chunk in a long section with the same heading vector.
        if section_title and (document_id, section_title) not in seen_sections:
            seen_sections.add((document_id, section_title))
            content = f"{section_title}\n\n{raw_content}"
        else:
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


# Matches a ToC entry line: any text followed by 2+ dots/spaces and a page number.
_TOC_LINE_RE = re.compile(r".*?[\s.]{2,}\d{1,4}\s*$")
# Matches a ToC entry line where pypdf collapses dot-leaders to a single space
# (e.g. "6 Conclusion, Limitations, and Future Directions 44").
_TOC_LINE_COMPACT_RE = re.compile(r".+\s\d{1,4}\s*$")


def _is_toc_page(page_text: str) -> bool:
    """Return True when the page looks like a Table of Contents or index page.

    A ToC page is identified by having at least 5 non-empty lines with a
    majority (≥50%) of them ending in optional dot-leaders/whitespace followed
    by a page number.  This catches both the standard "Title . . . . 28" form
    and the collapsed "Title 28" form that pypdf produces when dot-leaders are
    reduced to a single space.
    """
    non_empty = [l.strip() for l in page_text.splitlines() if l.strip()]
    if len(non_empty) < 5:
        return False
    toc_lines = sum(
        1 for l in non_empty
        if _TOC_LINE_RE.match(l) or _TOC_LINE_COMPACT_RE.match(l)
    )
    return toc_lines / len(non_empty) >= 0.5


def _segment_into_sections(page_text: str) -> list[tuple[str | None, str]]:
    """Split page_text at detected heading lines.

    Returns a list of (section_title, body) pairs. Text before the first
    heading has section_title=None. Headings must be ≤80 chars and must not
    end with sentence punctuation (.!?), to avoid false positives.

    ToC/index pages are returned as a single untitled segment to prevent
    each entry from being cloned under a false section_title.
    """
    if _is_toc_page(page_text):
        return [(None, page_text)]

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
            if not body:
                continue
            meta = {**base_meta, "page_number": page_idx}
            if section_title:
                meta["section_title"] = section_title
            documents.append(make_document(body, meta=meta))
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


def _ocr_pages(
    path: Path,
    session_id: str,
    covered_pages: set[int],
    ocr_language: str,
    ocr_workers: int = 1,
) -> list[Document]:
    try:
        import pytesseract
        from pypdf import PdfReader
    except ImportError:
        logger.warning(f"pytesseract not installed; skipping OCR for {path.name}")
        return []

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.warning(f"Cannot open {path} for OCR: {exc}")
        return []

    # Pre-collect images serially so PdfReader is only accessed from one thread.
    page_images: list[tuple[int, list]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        if page_idx in covered_pages:
            continue
        try:
            imgs = list(page.images)
        except Exception:
            continue
        if imgs:
            page_images.append((page_idx, imgs))

    if not page_images:
        return []

    source_meta = _source_metadata(path, session_id=session_id)
    results: list[Document | None] = [None] * len(page_images)
    tess_missing = threading.Event()

    def _ocr_one_page(slot: int, page_idx: int, imgs: list) -> None:
        page_texts: list[str] = []
        for img_file in imgs:
            if tess_missing.is_set():
                return
            try:
                text = pytesseract.image_to_string(img_file.image, lang=ocr_language)
                if text.strip():
                    page_texts.append(text.strip())
            except pytesseract.TesseractNotFoundError:
                # Tesseract binary not on PATH — disable OCR for this run.
                # Install: brew install tesseract tesseract-lang
                tess_missing.set()
                return
            except Exception as exc:
                logger.debug(f"OCR failed for image on page {page_idx} of {path.name}: {exc}")
        if page_texts:
            results[slot] = make_document(
                "\n".join(page_texts),
                meta={**source_meta, "page_number": page_idx, "extraction": "ocr"},
            )

    workers = max(1, ocr_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_ocr_one_page, i, page_idx, imgs)
            for i, (page_idx, imgs) in enumerate(page_images)
        ]
        for future in as_completed(futures):
            future.result()

    if tess_missing.is_set():
        logger.warning("Tesseract binary not found; OCR disabled for this session")

    return [doc for doc in results if doc is not None]


def _parse_single_pdf(args: tuple[Path, str, bool, str, int, int]) -> list[Document]:
    path, session_id, ocr_enabled, ocr_language, ocr_workers, ocr_min_text_chars = args
    covered_pages: set[int] = set()
    try:
        from haystack.components.converters.pypdf import PyPDFToDocument

        converter = PyPDFToDocument()
        result = converter.run(sources=[path])
        text_docs = (
            _split_into_pages(document_content(result["documents"][0]), str(path), session_id)
            if result["documents"]
            else []
        )
        if result["documents"] and ocr_min_text_chars > 0:
            full_text = document_content(result["documents"][0])
            for page_idx, raw in enumerate(full_text.split("\x0c"), start=1):
                if len(_clean_page_text(raw)) >= ocr_min_text_chars:
                    covered_pages.add(page_idx)
    except Exception as exc:
        logger.warning(f"Haystack PDF converter failed for {path}, using fallback: {exc}")
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = [(idx, page.extract_text() or "") for idx, page in enumerate(reader.pages, start=1)]
        # Reuse the same form-feed-joined splitter as the happy path so section
        # detection (section_title metadata) is applied in the fallback too.
        full_text = "\x0c".join(text for _, text in pages_text)
        text_docs = _split_into_pages(full_text, str(path), session_id)
        if ocr_min_text_chars > 0:
            for page_idx, text in pages_text:
                if len(_clean_page_text(text)) >= ocr_min_text_chars:
                    covered_pages.add(page_idx)

    if not ocr_enabled:
        return text_docs

    # OCR only pages not covered by pypdf text extraction.
    # For text-only PDFs page.images is empty anyway → no overhead.
    # For slide/mixed PDFs this captures body content that pypdf misses.
    return text_docs + _ocr_pages(path, session_id, covered_pages, ocr_language, ocr_workers)


def _load_pdf_documents(
    files: list[Path],
    *,
    session_id: str = DEFAULT_SESSION_ID,
    ocr_enabled: bool = True,
    ocr_language: str = "deu+eng",
    ocr_workers: int = 0,
    ocr_min_text_chars: int = 100,
) -> list[Document]:
    if not files:
        return []
    cpu = os.cpu_count() or 1
    effective_ocr_workers = ocr_workers if ocr_workers > 0 else cpu
    if len(files) == 1:
        return _parse_single_pdf((files[0], session_id, ocr_enabled, ocr_language, effective_ocr_workers, ocr_min_text_chars))
    max_workers = min(len(files), cpu)
    # Distribute OCR threads across file processes to avoid oversubscribing cores.
    per_file_ocr_workers = max(1, cpu // len(files))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_parse_single_pdf, (f, session_id, ocr_enabled, ocr_language, per_file_ocr_workers, ocr_min_text_chars))
            for f in files
        ]
    documents = []
    for f, future in zip(files, futures):
        try:
            documents.extend(future.result())
        except Exception as exc:
            logger.warning(f"Failed to parse PDF {f}: {exc}")
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

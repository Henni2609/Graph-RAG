from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from kg_rag.compat import Document, document_content, document_meta, make_document
from kg_rag.components.entity_extractor import EntityExtractor
from kg_rag.config import RagConfig
from kg_rag.logging import logger
from kg_rag.neo4j_store import Neo4jGraphStore, stable_id


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
            hf_config=config.hf,
            max_tokens=config.entity_max_tokens,
        )

    def run(self, paths: Iterable[str | Path], *, overwrite: bool = False) -> int:
        files = collect_supported_files(paths)
        logger.info(f"Indexing {len(files)} file(s)")
        if not files:
            return 0

        documents = load_documents(files)
        chunks = split_documents(
            documents,
            split_length=self.config.chunk_split_length,
            split_overlap=self.config.chunk_split_overlap,
        )
        chunks = normalize_chunk_metadata(chunks)
        embedded_chunks = embed_documents(chunks, model=self.config.embedding_model)
        enriched_chunks = self.entity_extractor.run(embedded_chunks)["documents"]
        self.store.setup_schema()
        self.store.persist_documents(enriched_chunks, overwrite=overwrite)
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


def load_documents(files: list[Path]) -> list[Document]:
    text_files = [path for path in files if path.suffix.lower() in {".txt", ".md"}]
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
    documents: list[Document] = []
    documents.extend(_load_text_documents(text_files))
    documents.extend(_load_pdf_documents(pdf_files))
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

        cleaner = DocumentCleaner()
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


def normalize_chunk_metadata(chunks: list[Document]) -> list[Document]:
    counters: dict[str, int] = defaultdict(int)
    normalized: list[Document] = []
    for chunk in chunks:
        meta = document_meta(chunk)
        source = str(meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown")
        document_id = str(meta.get("document_id") or stable_id(source))
        chunk_index = int(meta.get("chunk_index", meta.get("split_idx", counters[document_id])) or 0)
        counters[document_id] = max(counters[document_id], chunk_index + 1)
        meta.update(
            {
                "source": source,
                "title": meta.get("title") or Path(source).name,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "chunk_id": meta.get("chunk_id")
                or stable_id(f"{document_id}:{chunk_index}:{document_content(chunk)}"),
            }
        )
        if getattr(chunk, "meta", None) is not None:
            chunk.meta = meta
            normalized.append(chunk)
        else:
            normalized.append(make_document(document_content(chunk), meta=meta))
    return normalized


def fallback_sentence_split(
    documents: list[Document],
    *,
    split_length: int,
    split_overlap: int,
) -> list[Document]:
    import re

    chunks: list[Document] = []
    step = max(1, split_length - split_overlap)
    for document in documents:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", document_content(document)) if part.strip()]
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


def _load_text_documents(files: list[Path]) -> list[Document]:
    if not files:
        return []
    try:
        from haystack.components.converters.txt import TextFileToDocument

        converter = TextFileToDocument()
        result = converter.run(sources=files)
        documents = result["documents"]
        for index, document in enumerate(documents):
            source = _converted_source(document, files, index)
            _attach_source_metadata(document, source)
        return documents
    except Exception as exc:
        logger.warning(f"Haystack text converter unavailable, using fallback loader: {exc}")
        return [
            make_document(
                path.read_text(encoding="utf-8"),
                meta=_source_metadata(path),
            )
            for path in files
        ]


def _load_pdf_documents(files: list[Path]) -> list[Document]:
    if not files:
        return []
    try:
        from haystack.components.converters.pypdf import PyPDFToDocument

        converter = PyPDFToDocument()
        result = converter.run(sources=files)
        documents = result["documents"]
        for index, document in enumerate(documents):
            source = _converted_source(document, files, index)
            _attach_source_metadata(document, source)
        return documents
    except Exception as exc:
        logger.warning(f"Haystack PDF converter unavailable, using fallback loader: {exc}")
        from pypdf import PdfReader

        documents = []
        for path in files:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append(make_document(text, meta=_source_metadata(path)))
        return documents


def _attach_source_metadata(document: Document, source: str) -> None:
    meta = document_meta(document)
    meta.update(_source_metadata(Path(source)))
    document.meta = meta


def _converted_source(document: Document, files: list[Path], index: int) -> str:
    meta = document_meta(document)
    for key in ("source", "file_path", "path"):
        if meta.get(key):
            return str(meta[key])
    if index < len(files):
        return str(files[index])
    return "unknown"


def _source_metadata(path: Path) -> dict[str, str]:
    source = str(path.expanduser().resolve())
    return {
        "source": source,
        "title": path.name,
        "document_id": stable_id(source),
    }

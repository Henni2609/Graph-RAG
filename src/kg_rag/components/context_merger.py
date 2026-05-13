from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from kg_rag.compat import Document, component, document_content, document_meta

_CITATION_TAG = re.compile(r"\[S(\d+)\]")


@component
class ContextMerger:
    def __init__(self, max_context_chars: int = 6000) -> None:
        self.max_context_chars = max_context_chars

    @component.output_types(merged_context=str, documents=list[Document], citations=list[dict])
    def run(
        self,
        vector_docs: list[Document],
        graph_docs: list[Document],
        entity_context: str = "",
    ) -> dict[str, Any]:
        merged_documents = self.merge_documents(vector_docs, graph_docs)
        merged_context, citations = self.format_context(merged_documents, entity_context)
        return {
            "merged_context": merged_context,
            "documents": merged_documents,
            "citations": citations,
        }

    def merge_documents(
        self,
        vector_docs: list[Document],
        graph_docs: list[Document],
    ) -> list[Document]:
        seen: set[str] = set()
        merged: list[Document] = []

        for source, docs in (("vector", vector_docs), ("graph", graph_docs)):
            for document in docs:
                key = chunk_key(document)
                if key in seen:
                    continue
                seen.add(key)
                meta = document_meta(document)
                meta.setdefault("retrieval_source", source)
                if getattr(document, "meta", None) is not None:
                    document.meta = meta
                merged.append(document)

        return merged

    def format_context(
        self,
        documents: list[Document],
        entity_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        sections: list[str] = []
        citations: list[dict[str, Any]] = []
        if entity_context.strip():
            sections.append("[Entity-Relationen]\n" + entity_context.strip())

        for index, document in enumerate(documents, start=1):
            meta = document_meta(document)
            retrieval_source = meta.get("retrieval_source", "vector")
            label = "Semantisch relevant" if retrieval_source == "vector" else "Via Graph-Traversal"
            title = meta.get("title") or Path(str(meta.get("source", "unknown"))).name
            chunk_index = meta.get("chunk_index", "?")
            try:
                page_number = int(meta.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            page_number = max(1, page_number)
            text = document_content(document).strip()
            section = (
                f"[S{index}] [{label}] {title} · Seite {page_number} · Abschnitt {chunk_index}\n"
                f"{text}"
            )
            sections.append(section)
            citations.append(
                {
                    "index": index,
                    "document_id": meta.get("document_id"),
                    "title": title,
                    "page_number": page_number,
                    "chunk_index": chunk_index if isinstance(chunk_index, int) else None,
                    "snippet": text[:240],
                }
            )

        context = "\n\n".join(section for section in sections if section.strip())
        if len(context) <= self.max_context_chars:
            return context, citations
        truncated = context[: self.max_context_chars].rsplit("\n\n", 1)[0].strip()
        kept_indexes = {
            int(match.group(1))
            for match in _CITATION_TAG.finditer(truncated)
        }
        filtered = [c for c in citations if c["index"] in kept_indexes]
        return truncated, filtered


def chunk_key(document: Document) -> str:
    meta = document_meta(document)
    chunk_id = meta.get("chunk_id") or getattr(document, "id", None)
    if chunk_id:
        return str(chunk_id)
    digest = hashlib.sha256(document_content(document).encode("utf-8")).hexdigest()
    return digest

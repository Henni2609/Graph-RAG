from __future__ import annotations

import hashlib
from typing import Any

from kg_rag.compat import Document, component, document_content, document_meta


@component
class ContextMerger:
    def __init__(self, max_context_chars: int = 6000) -> None:
        self.max_context_chars = max_context_chars

    @component.output_types(merged_context=str, documents=list[Document])
    def run(
        self,
        vector_docs: list[Document],
        graph_docs: list[Document],
        entity_context: str = "",
    ) -> dict[str, Any]:
        merged_documents = self.merge_documents(vector_docs, graph_docs)
        merged_context = self.format_context(merged_documents, entity_context)
        return {"merged_context": merged_context, "documents": merged_documents}

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

    def format_context(self, documents: list[Document], entity_context: str) -> str:
        sections: list[str] = []
        if entity_context.strip():
            sections.append("[Entity-Relationen]\n" + entity_context.strip())

        for document in documents:
            meta = document_meta(document)
            retrieval_source = meta.get("retrieval_source", "vector")
            label = "Semantisch relevant" if retrieval_source == "vector" else "Via Graph-Traversal"
            chunk_id = meta.get("chunk_id") or getattr(document, "id", "unknown")
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", "?")
            section = (
                f"[{label}] chunk_id={chunk_id} source={source} chunk_index={chunk_index}\n"
                f"{document_content(document).strip()}"
            )
            sections.append(section)

        context = "\n\n".join(section for section in sections if section.strip())
        if len(context) <= self.max_context_chars:
            return context
        return context[: self.max_context_chars].rsplit("\n\n", 1)[0].strip()


def chunk_key(document: Document) -> str:
    meta = document_meta(document)
    chunk_id = meta.get("chunk_id") or getattr(document, "id", None)
    if chunk_id:
        return str(chunk_id)
    digest = hashlib.sha256(document_content(document).encode("utf-8")).hexdigest()
    return digest

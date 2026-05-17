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
                meta["relevance"] = getattr(document, "score", None)
                if getattr(document, "meta", None) is not None:
                    document.meta = meta
                merged.append(document)

        def _sort_key(doc: Document) -> tuple:
            m = document_meta(doc)
            doc_id = str(m.get("document_id") or "")
            idx = m.get("chunk_index")
            return (doc_id, idx if isinstance(idx, int) else 999999)

        merged.sort(key=_sort_key)
        return merged

    def format_context(
        self,
        documents: list[Document],
        entity_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        entity_section = ""
        if entity_context.strip():
            entity_section = "[Entity-Relationen]\n" + entity_context.strip()

        _SEP = 2  # len("\n\n")

        # Build per-document records: (position, header_body, text, citation, source, relevance)
        records: list[tuple[int, str, str, dict[str, Any], str, float | None]] = []
        for pos, document in enumerate(documents):
            meta = document_meta(document)
            retrieval_source = str(meta.get("retrieval_source", "vector"))
            relevance = meta.get("relevance")
            label = "Semantisch relevant" if retrieval_source == "vector" else "Via Graph-Traversal"
            title = meta.get("title") or Path(str(meta.get("source", "unknown"))).name
            chunk_index = meta.get("chunk_index", "?")
            section_title = meta.get("section_title", "")
            try:
                page_number = int(meta.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            page_number = max(1, page_number)
            text = document_content(document).strip()

            header_body = f"[{label}] {title} · Seite {page_number} · Abschnitt {chunk_index}"
            if section_title:
                header_body += f" · {section_title}"

            citation: dict[str, Any] = {
                "index": -1,
                "document_id": meta.get("document_id"),
                "title": title,
                "page_number": page_number,
                "chunk_index": chunk_index if isinstance(chunk_index, int) else None,
                "snippet": text,
            }
            records.append((pos, header_body, text, citation, retrieval_source, relevance))

        entity_overhead = len(entity_section) + _SEP if entity_section else 0
        budget = max(0, self.max_context_chars - entity_overhead)

        kept = _select_by_relevance(records, budget, _SEP)

        # Render kept records in position order; insert gap marker between non-consecutive chunks.
        parts: list[str] = []
        if entity_section:
            parts.append(entity_section)

        new_s = 1
        new_citations: list[dict[str, Any]] = []

        for _pos, header_body, text, cit, _src, _rel in kept:
            parts.append(f"[S{new_s}] {header_body}\n{text}")
            new_citations.append({**cit, "index": new_s})
            new_s += 1

        context = "\n\n".join(p for p in parts if p.strip())
        return context, new_citations


def _select_by_relevance(
    records: list[tuple],  # (pos, header_body, text, cit, source, relevance)
    budget: int,
    sep: int,
) -> list[tuple]:
    """Select records to fit within budget, ordered by cosine score descending."""
    if not records:
        return []

    costs = [len(hdr) + len(txt) + sep for _, hdr, txt, *_ in records]
    if sum(costs) <= budget:
        return records

    # All chunks sorted by cosine score desc — graph chunks now have real scores too.
    all_sorted = sorted(enumerate(records), key=lambda x: -(x[1][5] or 0.0))

    kept_idx: set[int] = set()
    used = 0
    for i, _rec in all_sorted:
        cost = costs[i]
        if used + cost > budget:
            continue
        kept_idx.add(i)
        used += cost

    # Return in ascending position order so context mirrors document structure.
    return [records[i] for i in sorted(kept_idx)]


def chunk_key(document: Document) -> str:
    meta = document_meta(document)
    chunk_id = meta.get("chunk_id") or getattr(document, "id", None)
    if chunk_id:
        return str(chunk_id)
    digest = hashlib.sha256(document_content(document).encode("utf-8")).hexdigest()
    return digest

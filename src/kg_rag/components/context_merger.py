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

        doc_sections: list[str] = []
        all_citations: list[dict[str, Any]] = []

        for index, document in enumerate(documents, start=1):
            meta = document_meta(document)
            retrieval_source = meta.get("retrieval_source", "vector")
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

            header = f"[S{index}] [{label}] {title} · Seite {page_number} · Abschnitt {chunk_index}"
            if section_title:
                header += f" · {section_title}"
            doc_sections.append(f"{header}\n{text}")
            all_citations.append(
                {
                    "index": index,
                    "document_id": meta.get("document_id"),
                    "title": title,
                    "page_number": page_number,
                    "chunk_index": chunk_index if isinstance(chunk_index, int) else None,
                    "snippet": text[:240],
                }
            )

        # Keep head + tail, drop the middle when over budget.
        # Entity section is fixed overhead, always kept.
        entity_overhead = len(entity_section) + 2 if entity_section else 0
        budget = max(0, self.max_context_chars - entity_overhead)

        kept_sections, kept_citations = _trim_middle(doc_sections, all_citations, budget)

        parts: list[str] = []
        if entity_section:
            parts.append(entity_section)
        parts.extend(kept_sections)
        context = "\n\n".join(p for p in parts if p.strip())
        return context, kept_citations


def _trim_middle(
    sections: list[str],
    citations: list[dict],
    budget: int,
) -> tuple[list[str], list[dict]]:
    """Keep sections from the head and tail; drop the middle when over budget."""
    sep = 2  # len("\n\n")
    if not sections:
        return [], []
    total = sum(len(s) + sep for s in sections) - sep
    if total <= budget:
        return sections, citations

    marker = "[… gekürzt …]"
    # Budget for actual content: subtract marker + its two surrounding separators.
    usable = budget - len(marker) - sep * 2
    if usable <= 0:
        return [], []

    # Split evenly: half for head (beginning of document), half for tail (end of document).
    # Tail gets any odd char so conclusions are never accidentally dropped.
    head_budget = usable // 2
    tail_budget = usable - head_budget

    head: list[str] = []
    head_idx: list[int] = []
    hc = 0
    for i, s in enumerate(sections):
        cost = len(s) + (sep if head else 0)
        if hc + cost > head_budget:
            break
        head.append(s)
        head_idx.append(i)
        hc += cost

    tail: list[str] = []
    tail_idx: list[int] = []
    tc = 0
    for i in range(len(sections) - 1, len(head) - 1, -1):
        cost = len(sections[i]) + (sep if tail else 0)
        if tc + cost > tail_budget:
            break
        tail.insert(0, sections[i])
        tail_idx.insert(0, i)
        tc += cost

    kept_sections = head + [marker] + tail
    kept_idx = set(head_idx) | set(tail_idx)
    kept_citations = [c for i, c in enumerate(citations) if i in kept_idx]
    return kept_sections, kept_citations


def chunk_key(document: Document) -> str:
    meta = document_meta(document)
    chunk_id = meta.get("chunk_id") or getattr(document, "id", None)
    if chunk_id:
        return str(chunk_id)
    digest = hashlib.sha256(document_content(document).encode("utf-8")).hexdigest()
    return digest

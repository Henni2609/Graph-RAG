from __future__ import annotations

import functools

from kg_rag.compat import Document, document_content, document_meta


@functools.lru_cache(maxsize=None)
def _get_cross_encoder(model: str, device: str = "cpu"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model, device=device)


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", top_k: int = 20, device: str = "cpu") -> None:
        self.model = model
        self.top_k = top_k
        self.device = device

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        try:
            encoder = _get_cross_encoder(self.model, self.device)
        except Exception:
            return documents[: self.top_k]
        pairs = [(query, document_content(doc)) for doc in documents]
        raw_scores = encoder.predict(pairs, show_progress_bar=False)
        scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        ranked = sorted(zip(scores, documents), key=lambda x: -x[0])
        result = []
        for score, doc in ranked[: self.top_k]:
            try:
                doc.score = float(score)
            except AttributeError:
                meta = document_meta(doc)
                meta["relevance"] = float(score)
                if hasattr(doc, "meta"):
                    doc.meta = meta
            result.append(doc)
        return result

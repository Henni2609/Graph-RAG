from __future__ import annotations

import functools
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from kg_rag.compat import Document, document_meta
from kg_rag.components.context_merger import ContextMerger
from kg_rag.components.entity_extractor import parse_extraction_response
from kg_rag.components.graph_retriever import GraphRetriever
from kg_rag.config import RagConfig
from kg_rag.llm import create_chat_generator, run_chat
from kg_rag.logging import log_timing, logger
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore


# Note: the UI renders [S1] tags as "Q1", "Q2" etc. (see decorateCitations in index.html).
# The LLM must output [S...] tags — they must match context headers exactly.
ANSWER_SYSTEM_PROMPT = """Du bist ein praeziser Graph-RAG-Assistent.
Beantworte die Frage auf Deutsch oder Englisch passend zur Sprache der Frage.
Antworte ausschliesslich auf Basis des bereitgestellten Kontexts. Verwende kein
eigenes Wissen und keine Informationen ausserhalb des Kontexts. Wenn relevante
Informationen im Kontext vorhanden sind, beantworte die Frage daraus — auch wenn
kein expliziter Abschnitt mit passendem Titel existiert. Wenn der Kontext die Frage
nicht oder nur teilweise abdeckt, sage das klar und antworte nur soweit der Kontext
reicht. Erfinde keine Fakten.

Laenge der Antwort:
Antworte so kurz wie moeglich und so ausfuehrlich wie noetig, um die Frage gut zu beantworten.
Keine Einleitungssaetze, keine Wiederholungen, kein Auffuellen.

Formatierung:
Nutze Fett (**...**) fuer wichtige Begriffe oder Schluesselaussagen.
Nutze Listen (-) wenn mehrere gleichwertige Punkte aufgezaehlt werden oder Schritte beschrieben werden.
Vermeide Codebloecke.

Zitiere Quellen ausschliesslich mit den Kurz-Tags [S1], [S2] usw., die am Anfang jedes
Kontextabschnitts stehen. Schreibe niemals (Dateiname, ...) oder Hex-Strings als Quellenangabe.
Zitiere jeden Abschnitt, aus dem du Informationen nutzt — nicht nur einen. Wenn mehrere
Abschnitte relevant sind, nenne alle. Zitiere jede Quelle als eigenen Tag, also [S1] [S3].
Niemals [S1, S3] oder [S1,S3]. Erfinde keine Tag-Nummern jenseits der vorhandenen Kontextabschnitte."""


QUERY_ENTITY_SYSTEM_PROMPT = """Extrahiere Entitaeten aus der Nutzerfrage.
Antworte ausschliesslich mit JSON:
{"entities":[{"name":"...","type":"Konzept","description":""}],"relations":[]}"""

_INSUFFICIENT_CONTEXT = (
    "Der bereitgestellte Kontext reicht nicht aus, um diese Frage zu beantworten."
)
_GENERATION_ERROR = (
    "Die Antwortgenerierung ist fehlgeschlagen. Bitte versuche es erneut."
)

# Per-session cache for embedding meta (model name + dimensions stored at index time).
# Avoids a Neo4j roundtrip on every query. Invalidated when a new index job completes
# or the session ends.
_embedding_meta_cache: dict[str, dict[str, Any]] = {}
_embedding_meta_lock = threading.Lock()


def invalidate_embedding_meta(session_id: str) -> None:
    with _embedding_meta_lock:
        _embedding_meta_cache.pop(session_id, None)


@dataclass
class QueryResult:
    answer: str
    context: str
    vector_documents: list[Document]
    graph_documents: list[Document]
    entity_context: str
    query_entities: list[str]
    citations: list[dict[str, Any]]


@dataclass
class RetrievalResult:
    context: str
    vector_documents: list[Document]
    graph_documents: list[Document]
    entity_context: str
    query_entities: list[str]
    citations: list[dict[str, Any]]
    query_embedding: list[float]
    early_answer: str | None


class QueryPipeline:
    def __init__(
        self,
        config: RagConfig,
        *,
        store: Neo4jGraphStore | None = None,
        generator: Any | None = None,
        merger: ContextMerger | None = None,
    ) -> None:
        self.config = config
        self.store = store or Neo4jGraphStore(config.neo4j)
        self.generator = generator
        self._extraction_gen: Any | None = None
        self.merger = merger or ContextMerger(max_context_chars=config.max_context_chars)
        self.graph_retriever = GraphRetriever(
            store=self.store,
            hops=config.graph_hops,
            limit=config.graph_limit,
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        hops: int | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> RetrievalResult:
        """Run all retrieval steps and return the data needed for answer generation."""
        self._check_embedding_compatibility(session_id)
        extraction_gen = self._extraction_generator()

        with log_timing("embedding"):
            query_embedding = embed_query(
                question, model=self.config.embedding_model, device=self.config.embedding_device
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            entity_future = pool.submit(
                self.extract_query_entities, question, generator=extraction_gen
            )
            with log_timing("vector_search"):
                try:
                    vector_documents = self.store.vector_search(
                        query_embedding,
                        top_k=top_k if top_k is not None else self.config.query_top_k,
                        session_id=session_id,
                    )
                except Exception as exc:
                    logger.error(f"Vector search failed: {exc}", exc_info=True)
                    return RetrievalResult(
                        context="",
                        vector_documents=[],
                        graph_documents=[],
                        entity_context="",
                        query_entities=[],
                        citations=[],
                        query_embedding=query_embedding,
                        early_answer="Ein Datenbankfehler ist aufgetreten. Bitte versuche es erneut.",
                    )
            with log_timing("entity_extraction_wait"):
                query_entities = entity_future.result()

        chunk_ids = [
            str(document_meta(document).get("chunk_id"))
            for document in vector_documents
            if document_meta(document).get("chunk_id")
        ]

        with log_timing("graph_search"):
            try:
                graph_result = self.graph_retriever.run(
                    chunk_ids=chunk_ids,
                    query_entities=query_entities,
                    query_embedding=query_embedding,
                    hops=hops,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error(f"Graph retrieval failed: {exc}", exc_info=True)
                graph_result = {"documents": [], "entity_context": ""}

        graph_documents = graph_result["documents"]
        entity_context = graph_result["entity_context"]

        # Only short-circuit when retrieval found literally nothing.
        # The system prompt handles the "context not sufficient" case for weak matches.
        if not vector_documents and not graph_documents and not entity_context.strip():
            return RetrievalResult(
                context="",
                vector_documents=vector_documents,
                graph_documents=[],
                entity_context=entity_context,
                query_entities=query_entities,
                citations=[],
                query_embedding=query_embedding,
                early_answer=_INSUFFICIENT_CONTEXT,
            )

        with log_timing("context_merge"):
            merge_result = self.merger.run(
                vector_docs=vector_documents,
                graph_docs=graph_documents,
                entity_context=entity_context,
            )

        return RetrievalResult(
            context=merge_result["merged_context"],
            vector_documents=vector_documents,
            graph_documents=graph_documents,
            entity_context=entity_context,
            query_entities=query_entities,
            citations=merge_result.get("citations", []),
            query_embedding=query_embedding,
            early_answer=None,
        )

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        hops: int | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> QueryResult:
        t0 = time.perf_counter()
        retrieval = self.retrieve(question, top_k=top_k, hops=hops, session_id=session_id)

        if retrieval.early_answer is not None:
            logger.info(f"TIMING total_query: {time.perf_counter() - t0:.3f}s (early exit)")
            return QueryResult(
                answer=retrieval.early_answer,
                context="",
                vector_documents=retrieval.vector_documents,
                graph_documents=retrieval.graph_documents,
                entity_context=retrieval.entity_context,
                query_entities=retrieval.query_entities,
                citations=[],
            )

        generator = self._generator()
        with log_timing("generate_answer"):
            answer = self.generate_answer(question, retrieval.context, generator=generator)
        answer = sanitize_citations(answer, {c["index"] for c in retrieval.citations})

        logger.info(f"TIMING total_query: {time.perf_counter() - t0:.3f}s")
        return QueryResult(
            answer=answer,
            context=retrieval.context,
            vector_documents=retrieval.vector_documents,
            graph_documents=retrieval.graph_documents,
            entity_context=retrieval.entity_context,
            query_entities=retrieval.query_entities,
            citations=retrieval.citations,
        )

    def extract_query_entities(self, question: str, *, generator: Any) -> list[str]:
        try:
            raw = run_chat(
                generator,
                QUERY_ENTITY_SYSTEM_PROMPT,
                question,
                generation_kwargs={
                    "temperature": 0,
                    "max_tokens": 200,
                    "timeout": 15,
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
        except Exception as exc:
            logger.warning(f"Query entity extraction failed: {exc}", exc_info=True)
            return []
        result = parse_extraction_response(raw)
        return [entity.name for entity in result.entities]

    def generate_answer(self, question: str, context: str, *, generator: Any) -> str:
        prompt = f"Kontext:\n{context}\n\nFrage:\n{question}"
        try:
            result = run_chat(
                generator,
                ANSWER_SYSTEM_PROMPT,
                prompt,
                generation_kwargs={
                    "temperature": 0.2,
                    "max_tokens": self.config.answer_max_tokens,
                    "timeout": self.config.answer_timeout_seconds,
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
        except Exception as exc:
            logger.warning(f"Answer generation failed: {exc}", exc_info=True)
            return _GENERATION_ERROR
        if not result:
            logger.warning("Answer LLM returned an empty reply")
            return _GENERATION_ERROR
        return result

    def _check_embedding_compatibility(self, session_id: str) -> None:
        with _embedding_meta_lock:
            cached = _embedding_meta_cache.get(session_id)
        if cached is None:
            try:
                stored = self.store.get_indexing_meta(session_id)
            except Exception:
                return
            if not stored:
                return
            with _embedding_meta_lock:
                _embedding_meta_cache[session_id] = stored
            cached = stored
        stored_model = cached.get("model", "")
        stored_dim = cached.get("dimensions")
        if stored_model and stored_model != self.config.embedding_model:
            raise RuntimeError(
                f"Embedding model mismatch: index was built with '{stored_model}' "
                f"but current config uses '{self.config.embedding_model}'. "
                "Please re-index your documents."
            )
        if stored_dim and stored_dim != self.config.embedding_dimensions:
            raise RuntimeError(
                f"Embedding dimension mismatch: index has {stored_dim} dims "
                f"but config specifies {self.config.embedding_dimensions}. "
                "Please re-index your documents."
            )

    def _generator(self) -> Any:
        if self.generator is None:
            self.generator = create_chat_generator(
                self.config.llm,
                timeout=self.config.answer_timeout_seconds,
                max_retries=self.config.answer_max_retries,
            )
        return self.generator

    def _extraction_generator(self) -> Any:
        if self._extraction_gen is None:
            self._extraction_gen = create_chat_generator(
                self.config.llm,
                model=self.config.llm.extraction_model,
                timeout=15,
                max_retries=1,
            )
        return self._extraction_gen

    def _error_result(self, message: str) -> QueryResult:
        return QueryResult(
            answer=message,
            context="",
            vector_documents=[],
            graph_documents=[],
            entity_context="",
            query_entities=[],
            citations=[],
        )


def sanitize_citations(answer: str, valid_indexes: set[int]) -> str:
    def _replace(m: re.Match) -> str:
        idx = int(m.group(1))
        return "" if idx not in valid_indexes else m.group(0)
    result = re.sub(r"\[S(\d+)\]", _replace, answer)
    return re.sub(r"  +", " ", result)


@functools.lru_cache(maxsize=None)
def _get_text_embedder(model: str, device: str) -> Any:
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.utils import ComponentDevice
    from kg_rag.pipelines.indexing import _resolve_device

    embedder = SentenceTransformersTextEmbedder(
        model=model,
        device=ComponentDevice.from_str(_resolve_device(device)),
    )
    embedder.warm_up()
    return embedder


def embed_query(question: str, *, model: str, device: str = "auto") -> list[float]:
    return _get_text_embedder(model, device).run(text=question)["embedding"]

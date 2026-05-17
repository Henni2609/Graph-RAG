from __future__ import annotations

import functools
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from kg_rag.compat import Document, document_meta
from kg_rag.components.context_merger import ContextMerger
from kg_rag.components.entity_extractor import parse_extraction_response
from kg_rag.components.graph_retriever import GraphRetriever
from kg_rag.config import RagConfig
from kg_rag.llm import create_chat_generator, run_chat
from kg_rag.logging import logger
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore


# Note: the UI renders [S1] tags as "Q1", "Q2" etc. (see decorateCitations in index.html).
# The LLM must output [S...] tags — they must match context headers exactly.
ANSWER_SYSTEM_PROMPT = """Du bist ein praeziser Graph-RAG-Assistent.
Beantworte die Frage auf Deutsch oder Englisch passend zur Sprache der Frage.
Nutze den bereitgestellten Kontext. Wenn relevante Informationen vorhanden sind, beantworte
die Frage daraus — auch wenn kein expliziter Abschnitt mit passendem Titel existiert.
Sage nur dann, dass du nicht antworten kannst, wenn der Kontext keinerlei relevante
Informationen enthaelt.

Struktur der Antwort:
Du bist frei mit der Struktur. Ueberlege dir ob Stichpunkte sinnvoll sind.

Zitiere Quellen ausschliesslich mit den Kurz-Tags [S1], [S2] usw., die am Anfang jedes
Kontextabschnitts stehen. Schreibe niemals (Dateiname, ...) oder Hex-Strings als Quellenangabe.
Zitiere jeden Abschnitt, aus dem du Informationen nutzt — nicht nur einen. Wenn mehrere
Abschnitte relevant sind, nenne alle. Zitiere jede Quelle als eigenen Tag, also [S1] [S3].
Niemals [S1, S3] oder [S1,S3]. Erfinde keine Tag-Nummern jenseits der vorhandenen Kontextabschnitte.

Nutze Markdown sparsam fuer Fett (**...**) und Listen (-). Vermeide Codebloecke, ueberlange
Antworten und ausschweifende Wiederholungen."""


QUERY_ENTITY_SYSTEM_PROMPT = """Extrahiere Entitaeten aus der Nutzerfrage.
Antworte ausschliesslich mit JSON:
{"entities":[{"name":"...","type":"Konzept","description":""}],"relations":[]}"""

_INSUFFICIENT_CONTEXT = (
    "Der bereitgestellte Kontext reicht nicht aus, um diese Frage zu beantworten."
)
_GENERATION_ERROR = (
    "Die Antwortgenerierung ist fehlgeschlagen. Bitte versuche es erneut."
)


@dataclass
class QueryResult:
    answer: str
    context: str
    vector_documents: list[Document]
    graph_documents: list[Document]
    entity_context: str
    query_entities: list[str]
    citations: list[dict[str, Any]]


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

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        hops: int | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> QueryResult:
        self._check_embedding_compatibility(session_id)

        generator = self._generator()
        extraction_gen = self._extraction_generator()

        with ThreadPoolExecutor(max_workers=1) as pool:
            entity_future = pool.submit(
                self.extract_query_entities, question, generator=extraction_gen
            )
            query_embedding = embed_query(question, model=self.config.embedding_model)
            try:
                vector_documents = self.store.vector_search(
                    query_embedding,
                    top_k=top_k if top_k is not None else self.config.query_top_k,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error(f"Vector search failed: {exc}", exc_info=True)
                return self._error_result(
                    "Ein Datenbankfehler ist aufgetreten. Bitte versuche es erneut."
                )
            query_entities = entity_future.result()

        chunk_ids = [
            str(document_meta(document).get("chunk_id"))
            for document in vector_documents
            if document_meta(document).get("chunk_id")
        ]

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
            return QueryResult(
                answer=_INSUFFICIENT_CONTEXT,
                context="",
                vector_documents=vector_documents,
                graph_documents=[],
                entity_context=entity_context,
                query_entities=query_entities,
                citations=[],
            )

        merge_result = self.merger.run(
            vector_docs=vector_documents,
            graph_docs=graph_documents,
            entity_context=entity_context,
        )
        context = merge_result["merged_context"]
        citations = merge_result.get("citations", [])

        answer = self.generate_answer(question, context, generator=generator)
        answer = sanitize_citations(answer, {c["index"] for c in citations})

        return QueryResult(
            answer=answer,
            context=context,
            vector_documents=vector_documents,
            graph_documents=graph_documents,
            entity_context=entity_context,
            query_entities=query_entities,
            citations=citations,
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
        try:
            stored = self.store.get_indexing_meta(session_id)
        except Exception:
            return
        if not stored:
            return
        stored_model = stored.get("model", "")
        stored_dim = stored.get("dimensions")
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
def _get_text_embedder(model: str) -> Any:
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    embedder = SentenceTransformersTextEmbedder(model=model)
    embedder.warm_up()
    return embedder


def embed_query(question: str, *, model: str) -> list[float]:
    return _get_text_embedder(model).run(text=question)["embedding"]

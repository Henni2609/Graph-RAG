from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_rag.compat import Document, document_meta
from kg_rag.components.context_merger import ContextMerger
from kg_rag.components.entity_extractor import parse_extraction_response
from kg_rag.components.graph_retriever import GraphRetriever
from kg_rag.config import RagConfig
from kg_rag.llm import create_chat_generator, run_chat
from kg_rag.neo4j_store import DEFAULT_SESSION_ID, Neo4jGraphStore


ANSWER_SYSTEM_PROMPT = """Du bist ein praeziser Graph-RAG-Assistent.
Beantworte die Frage auf Deutsch oder Englisch passend zur Sprache der Frage.
Nutze ausschliesslich den bereitgestellten Kontext. Wenn der Kontext nicht reicht, sage das klar.

Struktur der Antwort:
1. Beginne mit Saetzen, die das Konzept klar definieren: was es ist und wozu es dient.
2. Danach optional Stichpunkte oder kurze Absaetze mit relevanten Details.

Zitiere Quellen kompakt im Format (Dateiname, Abschnitt N) oder mit den Kurz-Tags [S1], [S2],
die im Kontext angegeben sind. Verwende niemals chunk_id-Hashes oder lange Hex-Strings.

Nutze Markdown sparsam fuer Fett (**...**) und Listen (-). Vermeide Codebloecke, ueberlange
Antworten und ausschweifende Wiederholungen."""


QUERY_ENTITY_SYSTEM_PROMPT = """Extrahiere Entitaeten aus der Nutzerfrage.
Antworte ausschliesslich mit JSON:
{"entities":[{"name":"...","type":"Konzept","description":""}],"relations":[]}"""


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
        generator = self._generator()
        query_entities = self.extract_query_entities(question, generator=generator)
        query_embedding = embed_query(question, model=self.config.embedding_model)

        vector_documents = self.store.vector_search(
            query_embedding,
            top_k=top_k if top_k is not None else self.config.query_top_k,
            session_id=session_id,
        )
        chunk_ids = [
            str(document_meta(document).get("chunk_id"))
            for document in vector_documents
            if document_meta(document).get("chunk_id")
        ]
        graph_result = self.graph_retriever.run(
            chunk_ids=chunk_ids,
            query_entities=query_entities,
            hops=hops,
            session_id=session_id,
        )
        graph_documents = graph_result["documents"]
        entity_context = graph_result["entity_context"]

        merge_result = self.merger.run(
            vector_docs=vector_documents,
            graph_docs=graph_documents,
            entity_context=entity_context,
        )
        context = merge_result["merged_context"]
        citations = merge_result.get("citations", [])
        answer = self.generate_answer(question, context, generator=generator)

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
                    "timeout": 4,
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
        except Exception:
            return []
        result = parse_extraction_response(raw)
        return [entity.name for entity in result.entities]

    def generate_answer(self, question: str, context: str, *, generator: Any) -> str:
        prompt = f"Kontext:\n{context}\n\nFrage:\n{question}"
        return run_chat(
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

    def _generator(self) -> Any:
        if self.generator is None:
            self.generator = create_chat_generator(
                self.config.llm,
                timeout=self.config.answer_timeout_seconds,
                max_retries=0,
            )
        return self.generator


def embed_query(question: str, *, model: str) -> list[float]:
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    embedder = SentenceTransformersTextEmbedder(model=model)
    embedder.warm_up()
    result = embedder.run(text=question)
    return result["embedding"]

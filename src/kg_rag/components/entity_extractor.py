from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from kg_rag.compat import Document, component, document_content, document_meta, make_document
from kg_rag.config import LLMConfig
from kg_rag.llm import create_chat_generator, run_chat
from kg_rag.logging import logger
from kg_rag.schema import Entity, ExtractionResult, Relation


ENTITY_SYSTEM_PROMPT = """Du extrahierst Knowledge-Graph-Daten aus Text-Chunks.
Antworte ausschliesslich mit gueltigem JSON im folgenden Format:
{
  "entities": [
    {"name": "...", "type": "Person|Org|Konzept|Location|Produkt|Technologie", "description": "..."}
  ],
  "relations": [
    {"source": "...", "target": "...", "relation": "VERWENDET|GEHOERT_ZU|ENTWICKELTE|RELATES_TO|..."}
  ]
}
Nutze nur Fakten, die direkt im Text stehen. Erfinde keine Entitaeten oder Relationen."""


def parse_extraction_response(raw_text: str) -> ExtractionResult:
    payload = _extract_json_object(raw_text)
    if not payload:
        return ExtractionResult()

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Could not parse entity extraction JSON")
        return ExtractionResult()

    entities: list[Entity] = []
    seen_entities: set[str] = set()
    for raw_entity in data.get("entities", []):
        if not isinstance(raw_entity, dict):
            continue
        entity = Entity.from_mapping(raw_entity)
        if not entity.name or entity.normalized_name() in seen_entities:
            continue
        seen_entities.add(entity.normalized_name())
        entities.append(entity)

    entity_names = {entity.name for entity in entities}
    normalized_entity_names = {entity.normalized_name() for entity in entities}
    relations: list[Relation] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for raw_relation in data.get("relations", []):
        if not isinstance(raw_relation, dict):
            continue
        relation = Relation.from_mapping(raw_relation)
        if not relation.source or not relation.target:
            continue
        if (
            relation.source not in entity_names
            and relation.source.casefold() not in normalized_entity_names
        ):
            continue
        if (
            relation.target not in entity_names
            and relation.target.casefold() not in normalized_entity_names
        ):
            continue
        key = (relation.source.casefold(), relation.target.casefold(), relation.relation.casefold())
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append(relation)

    return ExtractionResult(entities=entities, relations=relations)


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""


@component
class EntityExtractor:
    def __init__(
        self,
        generator: Any | None = None,
        llm_config: LLMConfig | None = None,
        max_tokens: int = 800,
        concurrency: int = 10,
    ) -> None:
        self.generator = generator
        self.llm_config = llm_config
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        *,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, list[Document]]:
        total = len(documents)
        if progress is not None:
            progress("extracting", 0, total)
        if not documents:
            return {"documents": []}

        generator = self._generator()
        workers = min(self.concurrency, total)

        def extract(document: Document) -> tuple[list[dict], list[dict]]:
            try:
                text = run_chat(
                    generator,
                    ENTITY_SYSTEM_PROMPT,
                    f"Text-Chunk:\n{document_content(document)}",
                    generation_kwargs={
                        "temperature": 0,
                        "max_tokens": self.max_tokens,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                )
                result = parse_extraction_response(text)
                entities = [entity.__dict__ for entity in result.entities]
                relations = [relation.__dict__ for relation in result.relations]
                return entities, relations
            except Exception as exc:
                logger.warning(f"Entity extraction failed for chunk: {exc}")
                return [], []

        results: list[tuple[list[dict], list[dict]] | None] = [None] * total
        done_count = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_index = {pool.submit(extract, doc): i for i, doc in enumerate(documents)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                done_count += 1
                if progress is not None:
                    progress("extracting", done_count, total)

        enriched_documents: list[Document] = []
        for document, payload in zip(documents, results):
            entities, relations = payload if payload is not None else ([], [])
            meta = document_meta(document)
            meta["entities"] = entities
            meta["relations"] = relations
            enriched_documents.append(
                make_document(
                    document_content(document),
                    meta=meta,
                    embedding=getattr(document, "embedding", None),
                    score=getattr(document, "score", None),
                    id=getattr(document, "id", None),
                )
            )

        return {"documents": enriched_documents}

    def _generator(self) -> Any:
        if self.generator is not None:
            return self.generator
        if self.llm_config is None:
            self.llm_config = LLMConfig.from_env()
        self.generator = create_chat_generator(
            self.llm_config,
            model=self.llm_config.extraction_model,
        )
        return self.generator

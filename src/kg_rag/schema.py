from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_ENTITY_TYPES = frozenset({"Person", "Org", "Konzept", "Location", "Produkt", "Technologie"})


@dataclass(frozen=True)
class Entity:
    name: str
    type: str = "Konzept"
    description: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Entity":
        raw_type = str(data.get("type", "Konzept")).strip()
        entity_type = raw_type if raw_type in _VALID_ENTITY_TYPES else "Konzept"
        return cls(
            name=str(data.get("name", "")).strip(),
            type=entity_type,
            description=str(data.get("description", "")).strip(),
        )

    def normalized_name(self) -> str:
        return normalize_entity_name(self.name)


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            source=str(data.get("source", "")).strip(),
            target=str(data.get("target", "")).strip(),
            relation=str(data.get("relation", "RELATES_TO")).strip() or "RELATES_TO",
        )


@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


def normalize_entity_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())

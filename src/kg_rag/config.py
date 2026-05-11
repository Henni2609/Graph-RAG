from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class HuggingFaceConfig:
    api_token: str
    provider: str = ""
    generation_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    entity_extraction_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    base_url: str = HF_ROUTER_BASE_URL

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        return cls(
            api_token=os.getenv("HF_API_TOKEN", ""),
            provider=os.getenv("HF_PROVIDER", ""),
            generation_model=os.getenv("GENERATION_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            entity_extraction_model=os.getenv("ENTITY_EXTRACTION_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            base_url=os.getenv("HF_BASE_URL", "") or HF_ROUTER_BASE_URL,
        )

    def routed_model(self) -> str:
        if self.provider:
            return f"{self.generation_model}:{self.provider}"
        return self.generation_model

    def require_runtime_values(self) -> None:
        if not self.api_token:
            raise RuntimeError("Missing required environment variable: HF_API_TOKEN")


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password123"
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password123"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass(frozen=True)
class RagConfig:
    hf: HuggingFaceConfig
    neo4j: Neo4jConfig
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_split_length: int = 10
    chunk_split_overlap: int = 2
    query_top_k: int = 5
    graph_hops: int = 2
    graph_limit: int = 8
    max_context_chars: int = 6000
    entity_max_tokens: int = 800
    answer_max_tokens: int = 1500

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            hf=HuggingFaceConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_split_length=_get_int("CHUNK_SPLIT_LENGTH", 10),
            chunk_split_overlap=_get_int("CHUNK_SPLIT_OVERLAP", 2),
            query_top_k=_get_int("QUERY_TOP_K", 5),
            graph_hops=_get_int("GRAPH_HOPS", 2),
            graph_limit=_get_int("GRAPH_LIMIT", 8),
            max_context_chars=_get_int("MAX_CONTEXT_CHARS", 6000),
            entity_max_tokens=_get_int("ENTITY_MAX_TOKENS", 800),
            answer_max_tokens=_get_int("ANSWER_MAX_TOKENS", 1500),
        )

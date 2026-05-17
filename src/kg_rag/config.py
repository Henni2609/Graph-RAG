from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-pro"
DEFAULT_EXTRACTION_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = DEFAULT_LLM_MODEL
    extraction_model: str = DEFAULT_EXTRACTION_MODEL
    base_url: str = DEEPSEEK_BASE_URL

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            extraction_model=os.getenv("LLM_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
            base_url=os.getenv("LLM_BASE_URL", "") or DEEPSEEK_BASE_URL,
        )

    def require_runtime_values(self) -> None:
        if not self.api_key:
            raise RuntimeError("Missing required environment variable: LLM_API_KEY")


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
    llm: LLMConfig
    neo4j: Neo4jConfig
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimensions: int = 384
    chunk_split_length: int = 10
    chunk_split_overlap: int = 2
    query_top_k: int = 8
    graph_hops: int = 2
    graph_max_hops: int = 3
    graph_limit: int = 8
    min_similarity: float = 0.25
    max_context_chars: int = 16000
    entity_max_tokens: int = 1200
    answer_max_tokens: int = 500
    answer_timeout_seconds: int = 60
    answer_max_retries: int = 2
    extraction_concurrency: int = 10

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            llm=LLMConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            embedding_dimensions=_get_int("EMBEDDING_DIMENSIONS", 384),
            chunk_split_length=_get_int("CHUNK_SPLIT_LENGTH", 10),
            chunk_split_overlap=_get_int("CHUNK_SPLIT_OVERLAP", 2),
            query_top_k=_get_int("QUERY_TOP_K", 8),
            graph_hops=_get_int("GRAPH_HOPS", 2),
            graph_max_hops=_get_int("GRAPH_MAX_HOPS", 3),
            graph_limit=_get_int("GRAPH_LIMIT", 8),
            min_similarity=_get_float("MIN_SIMILARITY", 0.25),
            max_context_chars=_get_int("MAX_CONTEXT_CHARS", 16000),
            entity_max_tokens=_get_int("ENTITY_MAX_TOKENS", 1200),
            answer_max_tokens=_get_int("ANSWER_MAX_TOKENS", 500),
            answer_timeout_seconds=_get_int("ANSWER_TIMEOUT_SECONDS", 60),
            answer_max_retries=_get_int("ANSWER_MAX_RETRIES", 2),
            extraction_concurrency=_get_int("EXTRACTION_CONCURRENCY", 10),
        )

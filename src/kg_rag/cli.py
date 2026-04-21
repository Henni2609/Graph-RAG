from __future__ import annotations

import argparse
import os
from pathlib import Path

from kg_rag.config import RagConfig
from kg_rag.logging import configure_logging, logger
from kg_rag.neo4j_store import Neo4jGraphStore
from kg_rag.pipelines.indexing import IndexingPipeline
from kg_rag.pipelines.query import QueryPipeline


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command == "setup-schema":
        setup_schema()
    elif args.command == "index":
        index(args)
    elif args.command == "query":
        query(args)
    else:
        parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg-rag")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup-schema", help="Create Neo4j constraints and vector index")

    index_parser = subparsers.add_parser("index", help="Index .txt, .md, and .pdf documents")
    index_parser.add_argument("paths", nargs="+", help="Files or directories to index")
    index_parser.add_argument("--overwrite", action="store_true", help="Clear Neo4j before indexing")

    query_parser = subparsers.add_parser("query", help="Ask a question against the Graph RAG system")
    query_parser.add_argument("question", help="Question in German or English")
    query_parser.add_argument("--top-k", type=int, default=None)
    query_parser.add_argument("--hops", type=int, default=None)
    query_parser.add_argument("--show-context", action="store_true")
    return parser


def setup_schema() -> None:
    config = RagConfig.from_env()
    store = Neo4jGraphStore(config.neo4j)
    try:
        store.setup_schema()
        logger.info("Neo4j schema is ready")
    finally:
        store.close()


def index(args: argparse.Namespace) -> None:
    config = RagConfig.from_env()
    pipeline = IndexingPipeline(config)
    try:
        chunk_count = pipeline.run(args.paths, overwrite=args.overwrite)
        logger.info(f"Indexed {chunk_count} chunk(s)")
    finally:
        pipeline.store.close()


def query(args: argparse.Namespace) -> None:
    config = RagConfig.from_env()
    pipeline = QueryPipeline(config)
    try:
        result = pipeline.run(args.question, top_k=args.top_k, hops=args.hops)
        print(result.answer)
        if args.show_context:
            print("\n--- Kontext ---")
            print(result.context)
    finally:
        pipeline.store.close()


def load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()

# Knowledge Graph RAG System

Python implementation of the PRD for a Haystack 2.x + Neo4j 5.x Graph RAG system using a HuggingFace Dedicated Endpoint for Gemma 4 31B Instruct.

## Features

- Offline ingestion for `.txt`, `.md`, and `.pdf`
- Sentence-based chunking with overlap
- `all-MiniLM-L6-v2` document/query embeddings
- LLM-based entity and relation extraction
- Neo4j graph schema with `Document`, `Chunk`, and `Entity` nodes
- Hybrid retrieval: vector search + graph traversal + entity context
- Structured answer generation prompt with source chunk references
- CLI for schema setup, indexing, and querying
- Docker Compose setup for local Neo4j

## Quickstart

```bash
cp .env.example .env
docker compose up -d neo4j
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
kg-rag setup-schema
kg-rag index ./docs --overwrite
kg-rag query "Welche Technologien nutzt das System?"
```

Set `HF_API_TOKEN` and `HF_ENDPOINT_URL` in `.env` before running indexing or queries that call the LLM.

## Environment

| Variable | Required | Description |
| --- | --- | --- |
| `HF_API_TOKEN` | yes | HuggingFace token for the Dedicated Endpoint |
| `HF_ENDPOINT_URL` | yes | Dedicated Endpoint URL |
| `NEO4J_URI` | yes | Neo4j Bolt URI |
| `NEO4J_USERNAME` | yes | Neo4j username |
| `NEO4J_PASSWORD` | yes | Neo4j password |
| `NEO4J_DATABASE` | no | Neo4j database, defaults to `neo4j` |
| `EMBEDDING_MODEL` | no | Defaults to `sentence-transformers/all-MiniLM-L6-v2` |

## CLI

```bash
kg-rag setup-schema
kg-rag index ./path/to/files --overwrite
kg-rag query "Explain the graph retrieval path" --top-k 5 --hops 2
```

`--overwrite` clears the existing graph before indexing. Without it, new documents are merged by source path.

## Architecture

Indexing:

1. Load files with Haystack converters.
2. Split into sentence chunks.
3. Embed chunks with `SentenceTransformersDocumentEmbedder`.
4. Extract entities and relations through `HuggingFaceAPIChatGenerator`.
5. Persist documents, chunks, embeddings, mentions, relations, and `NEXT_CHUNK` edges in Neo4j.

Query:

1. Embed query with `SentenceTransformersTextEmbedder`.
2. Retrieve top-k chunks through Neo4j vector search.
3. Traverse 1-3 graph hops from vector hits and recognized query entities.
4. Merge and de-duplicate context up to `MAX_CONTEXT_CHARS`.
5. Generate the final answer through the configured HuggingFace endpoint.

## Tests

```bash
pytest
```

The tests cover local parsing, context merging, and graph persistence query behavior with fakes. They do not require a HuggingFace token or Neo4j instance.

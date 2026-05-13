# Knowledge Graph RAG System

A self-hosted Retrieval-Augmented Generation system that indexes PDF and text documents into a Neo4j knowledge graph and answers questions using hybrid retrieval — vector similarity over chunks combined with multi-hop graph traversal over extracted entities.

## What it does

Given a corpus of documents, the system:

1. **Indexes** — splits each document into chunks, embeds them, and uses an LLM to extract typed entities and relations per chunk.
2. **Stores** — persists chunks (with embeddings), entities, and relations in Neo4j as a property graph with a native vector index.
3. **Retrieves** — answers questions by combining vector search over chunks with graph traversal over the entities the question and the top chunks mention.
4. **Generates** — feeds the merged context to an LLM and returns an answer that cites the source chunks it used.

The web UI lets you drag-and-drop PDFs and watch the knowledge graph grow.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Web UI (FastAPI + Uvicorn)                  │
│   GET  /              → static/index.html (vis-network SPA)      │
│   GET  /api/graph     → Neo4jGraphStore.fetch_entity_graph()     │
│   POST /api/upload    → IndexingPipeline.run()                   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ IndexingPipeline  │   │  QueryPipeline    │   │  CLI (kg-rag)     │
│  load → split →   │   │  embed question → │   │  setup-schema /   │
│  embed → extract  │   │  vector + graph + │   │  index / query /  │
│  → persist        │   │  merge → generate │   │  serve            │
└────────┬──────────┘   └─────────┬─────────┘   └─────────┬─────────┘
         │                        │                       │
         └──────────┬─────────────┼───────────────────────┘
                    ▼             ▼
        ┌───────────────────┐   ┌──────────────────────┐
        │  HuggingFace      │   │  Sentence-           │
        │  Inference        │   │  Transformers        │
        │  Providers        │   │  (MiniLM-L6-v2,      │
        │  (Llama 3.3 70B   │   │   384-dim, cosine)   │
        │  default, via     │   └──────────────────────┘
        │  OpenAI-compat    │
        │  router)          │              │
        └───────────────────┘              ▼
                                 ┌────────────────────────────┐
                                 │  Neo4jGraphStore           │
                                 │  - schema setup            │
                                 │  - persist documents       │
                                 │  - vector search           │
                                 │  - graph traversal         │
                                 │  - entity-graph dump       │
                                 └────────────┬───────────────┘
                                              │ Bolt
                                              ▼
                                 ┌────────────────────────────┐
                                 │  Neo4j 5+                  │
                                 │  - Property graph          │
                                 │  - Native vector index     │
                                 └────────────────────────────┘
```

### Stack

| Layer | Technology |
| --- | --- |
| LLM | DeepSeek API (OpenAI-compatible, `https://api.deepseek.com`). Default model: `deepseek-v4-pro`. Any OpenAI-compatible endpoint can be substituted via `LLM_BASE_URL`. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dim, cosine) |
| Pipeline components | Haystack 2.x (`OpenAIChatGenerator` pointed at the DeepSeek API; `SentenceTransformersDocumentEmbedder`, `DocumentSplitter`, etc.) |
| Graph DB | Neo4j 5+ |
| Web backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS + `vis-network` 9.x (loaded from CDN) |
| Language | Python 3.10+ |

## Graph schema

```
(:Document {id, title, source, created_at, updated_at})
   -[:HAS_CHUNK]→
(:Chunk {id, text, embedding[384], chunk_index, document_id, source, title})
   -[:NEXT_CHUNK]→ (:Chunk)
   -[:MENTIONS]→
(:Entity {name, name_normalized, type, description})
   -[:RELATES_TO {relation, chunk_id}]→ (:Entity)
```

**Uniqueness constraints**
- `Document.id`
- `Chunk.id`
- `Entity.name_normalized`

**Vector index**
- `chunk_embeddings` on `Chunk.embedding`, 384 dimensions, cosine similarity.

`name_normalized` is the casefolded version of an entity's surface form. It is the only key used for entity identity — aliases are not tracked.

## Data flow

### Indexing
Triggered via `kg-rag index <paths>` or `POST /api/upload`.

1. **Load** — Haystack converters (`TextFileToDocument`, `PyPDFToDocument`) read `.txt`, `.md`, `.pdf`. Fallback loaders are used if Haystack converters fail to import.
2. **Clean & split** — `DocumentCleaner` + `DocumentSplitter` produce sentence-based chunks with configurable length and overlap (default 10 / 2). A regex-based fallback handles environments without the Haystack splitter.
3. **Embed** — `SentenceTransformersDocumentEmbedder` embeds each chunk into a 384-dim vector.
4. **Extract** — for each chunk, the LLM is called with a strict JSON-only prompt that asks for `entities` (typed) and `relations` (source → target with a verb-like label). Calls run in parallel with a bounded thread pool (`EXTRACTION_CONCURRENCY`, default 10) and use the lighter `LLM_EXTRACTION_MODEL` (default `deepseek-v4-flash`). Malformed JSON returns an empty extraction. Relations whose source/target aren't in the chunk's entity list are dropped.
5. **Persist** — four batched `UNWIND` Cypher writes per indexing run, not per chunk:
   - chunks batch: `MERGE` Document + Chunk + `HAS_CHUNK` for ~100 chunks at once.
   - entities batch: `MERGE` each `Entity` by `(name_normalized, session_id)` and create the `MENTIONS` edge in a single pass.
   - relations batch: `MERGE` `Entity-[:RELATES_TO {relation, chunk_id}]->Entity`.
   - chunk-sequence batch: `MERGE` adjacent `Chunk-[:NEXT_CHUNK]->Chunk` pairs in `chunk_index` order.

### Query
Triggered via `kg-rag query <question>` (no web search UI yet).

1. **Question entities** — LLM extracts entity names from the question (JSON, same shape as indexing).
2. **Question embedding** — `SentenceTransformersTextEmbedder` produces a query vector.
3. **Vector search** — Neo4j returns the top-k chunks via `db.index.vector.queryNodes`.
4. **Graph search** — From the chunk IDs returned by vector search and from the normalized question-entity names, traverse `Chunk-[:MENTIONS]->Entity-[:RELATES_TO*1..h]-Entity<-[:MENTIONS]-Chunk` up to `h` hops (default 2, clamped to 1–3). Excludes seed chunks. Caps results at `GRAPH_LIMIT`.
5. **Entity context** — Collect direct `RELATES_TO` neighbors of the question entities as readable lines: `Source --relation--> Target`.
6. **Merge** — Combine vector chunks, graph chunks, and the entity-context block under a `MAX_CONTEXT_CHARS` budget. Deduplicate by `chunk_id`. Vector chunks rank first.
7. **Generate** — The LLM is prompted with the merged context and the question, and instructed to cite `chunk_id`s for every fact.

## Project structure

```
src/kg_rag/
├── __init__.py
├── cli.py                       # argparse entry point: setup-schema, index, query, serve
├── config.py                    # RagConfig + LLMConfig + Neo4jConfig
├── compat.py                    # Haystack-version compatibility shims (Document, @component, ...)
├── llm.py                       # create_chat_generator — Haystack OpenAIChatGenerator pointed at the DeepSeek API
├── logging.py                   # loguru setup
├── neo4j_store.py               # Neo4jGraphStore: schema, persist, vector_search, graph_search, entity_context, fetch_entity_graph
├── schema.py                    # Entity / Relation / ExtractionResult dataclasses + normalize_entity_name
├── components/
│   ├── entity_extractor.py      # @component — LLM-based JSON extraction with validation
│   ├── graph_retriever.py       # @component — graph_search + entity_context wrapper
│   └── context_merger.py        # @component — deduplication and character-budget merge
├── pipelines/
│   ├── indexing.py              # IndexingPipeline + helpers (load/split/embed/normalize)
│   └── query.py                 # QueryPipeline + QueryResult
└── web/
    ├── __init__.py
    ├── app.py                   # FastAPI factory + uvicorn runner
    └── static/
        └── index.html           # Drag-drop + vis-network visualization

tests/                           # tests use fakes throughout; no Neo4j / DeepSeek / network required
docs/                            # Example markdown to index
docker-compose.yml               # Optional Docker-based Neo4j service
pyproject.toml
.env.example
```

## Setup

### 1. Install and start Neo4j

The simplest path on macOS is Homebrew (no Desktop GUI required):

```bash
brew install neo4j
neo4j-admin dbms set-initial-password password123
brew services start neo4j
```

Verify: open http://localhost:7474 — the Neo4j Browser should load.

Stop and start with `brew services stop neo4j` / `brew services start neo4j`.

Alternatives: Neo4j Desktop (https://neo4j.com/download/), Neo4j Community ZIP, or `docker compose up -d neo4j` if you have Docker.

### 2. Get a DeepSeek API key

1. Create an API key at https://platform.deepseek.com/api_keys.
2. Top up credits on the same dashboard — the API is pay-as-you-go and rejects requests once the balance hits zero.
3. The default model is `deepseek-v4-pro`. Switch via `LLM_MODEL` (e.g. `deepseek-v4-flash` for cheaper/faster calls).

Any OpenAI-compatible endpoint can be used instead by setting `LLM_BASE_URL` and `LLM_MODEL`.

### 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs Haystack, the OpenAI SDK (used against the DeepSeek API), the Neo4j driver, sentence-transformers (which pulls PyTorch), FastAPI/Uvicorn, and the test toolchain.

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env`. The only required entry is `LLM_API_KEY`. Everything else has sensible defaults.

### 5. Create the Neo4j schema

```bash
kg-rag setup-schema
```

This adds the three uniqueness constraints and the vector index. It is idempotent.

## Usage

### CLI

```bash
kg-rag setup-schema                                  # Once per Neo4j instance
kg-rag index ./docs                                  # Recurse a directory
kg-rag index ./paper.pdf --overwrite                 # Clear the graph before indexing
kg-rag query "How is X related to Y?"                # Ask in German or English
kg-rag query "..." --top-k 5 --hops 2 --show-context # Print retrieved context too
kg-rag serve --host 127.0.0.1 --port 8000            # Start the web UI
```

`--overwrite` runs `MATCH (n) DETACH DELETE n` followed by `setup-schema` before indexing. Without it, new documents are merged into the existing graph.

### Web UI

```bash
kg-rag serve
```

Open http://127.0.0.1:8000/. The page has two tabs in the header:

- **Graph** — renders all entities and `RELATES_TO` edges via `vis-network`. Refreshes after every successful upload.
- **Chat** — single-turn Q&A against the indexed corpus. Each answer is shown with chips reporting how many vector chunks and graph chunks fed the context, plus the entities the LLM extracted from the question.

The sidebar with drag-and-drop PDF upload (`POST /api/upload`) is visible in both views. On any failure, the backend error is shown in the sidebar entry and the top-right status.

API endpoints:
- `GET /` — SPA
- `GET /api/graph` — `{nodes, edges}` (requires `X-Session-Id` header)
- `POST /api/upload` — multipart PDF. **Async**: returns `202` with `{job_id, filename, status: "queued", estimated_seconds}` immediately. The actual indexing runs on a background thread.
- `GET /api/jobs/{job_id}` — returns the current `JobState`: `{status, step, current, total, chunks_indexed, error, graph, …}`. Steps cycle through `parsing → splitting → embedding → extracting → persisting → done`. The frontend polls this every 700 ms while a job is running. Jobs are session-scoped (cross-session access returns 404).
- `POST /api/query` — JSON `{question, top_k?, hops?}` → `{answer, query_entities, vector_chunks, graph_chunks, context}`
- `POST /api/session/end` — JSON `{session_id}` clears the session's data from Neo4j

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_API_KEY` | — (required) | DeepSeek API key (or any OpenAI-compatible key when `LLM_BASE_URL` is overridden) |
| `LLM_MODEL` | `deepseek-v4-pro` | Model for answer generation and query-side entity extraction |
| `LLM_EXTRACTION_MODEL` | `deepseek-v4-flash` | Model used during indexing for per-chunk entity/relation extraction. Defaults to a cheaper/faster sibling |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible base URL. Override to use a different provider |
| `NEO4J_URI` | `bolt://localhost:7687` | |
| `NEO4J_USERNAME` | `neo4j` | |
| `NEO4J_PASSWORD` | `password123` | Must match the password you set with `neo4j-admin dbms set-initial-password` |
| `NEO4J_DATABASE` | `neo4j` | |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must produce 384-dim vectors to match the vector index |
| `CHUNK_SPLIT_LENGTH` | `10` | Sentences per chunk |
| `CHUNK_SPLIT_OVERLAP` | `2` | Sentence overlap |
| `QUERY_TOP_K` | `5` | Vector hits before graph expansion |
| `GRAPH_HOPS` | `2` | Graph traversal depth, clamped to 1–3 |
| `GRAPH_LIMIT` | `8` | Max graph chunks returned per query |
| `MAX_CONTEXT_CHARS` | `6000` | Char budget for the merged context block |
| `ENTITY_MAX_TOKENS` | `800` | LLM output cap for entity extraction |
| `ANSWER_MAX_TOKENS` | `500` | Output cap for the final answer. Sized so V4 Pro non-thinking (~60 chars/s) typically finishes in 7–11 s |
| `ANSWER_TIMEOUT_SECONDS` | `11` | Hard per-request timeout for the answer-generation LLM call. Together with the 4 s budget for query-side entity extraction this targets a ≤15 s total `/api/query` round-trip |
| `EXTRACTION_CONCURRENCY` | `10` | Parallel in-flight LLM calls during entity extraction. Raise for faster indexing if your provider tolerates it |

## Testing

```bash
pytest -q
```

The tests cover, with fakes throughout:
- Local parsing and chunk-metadata normalization (`test_indexing_helpers.py`)
- Context-merge dedup and budget (`test_context_merger.py`)
- Entity-extraction JSON parsing, including dropped malformed relations (`test_entity_extractor.py`)
- Neo4j writes — that `MERGE` statements for chunks, entities, relations, and `NEXT_CHUNK` are issued; that graph traversal clamps hops to 1–3 (`test_neo4j_store.py`)
- Web endpoints — graph payload shape, upload validation, HTML serving, query endpoint with mocked pipeline (`test_web_app.py`)

No Neo4j, no network, no LLM key needed.

## How it differs from a vanilla RAG

A vanilla RAG retrieves chunks by vector similarity and passes them to the LLM. This system adds two ingredients:

1. **Per-chunk knowledge-graph extraction.** Entities and typed relations are persisted as graph structure rather than free text. "What's connected to X?" becomes a 1-hop Cypher query.
2. **Hybrid retrieval.** Top-k chunks come from vector search; additional chunks are pulled in via graph traversal over the entities those chunks mention plus the entities recognized in the question itself. The two sets are merged under a character budget.

Trade-offs to be aware of:
- Indexing is more expensive — one LLM call per chunk.
- Retrieval quality depends on extraction quality. A weak model produces a sparse, noisy graph that helps less than vanilla vector search would.
- For corpora where most queries are simple lookups, the graph layer adds latency without payoff.

## Known limitations

- The `kg-rag serve` entry point clears the entire Neo4j database at startup (any in-flight indexing job would be lost on restart). This is convenient for the single-user dev workflow but unsafe for shared deployments.
- No re-ranking stage between vector results and the LLM.
- Entity identity uses `name_normalized` (casefold only). Aliases, plurals, and minor spelling variants are treated as distinct entities.
- The web UI's upload history is in-page only — it resets on reload.
- The chat panel is single-turn — no conversation history is sent back to the LLM.

## Operations cheatsheet

```bash
# Start everything
brew services start neo4j
.venv/bin/kg-rag serve

# Stop everything
brew services stop neo4j           # or leave running for next session
pkill -f "uvicorn.*kg_rag.web"

# Inspect the graph in the browser
open http://localhost:7474

# Reset the graph
kg-rag index ./docs --overwrite
```

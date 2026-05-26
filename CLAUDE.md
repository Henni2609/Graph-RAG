# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Graph RAG System

### Stack

| Schicht | Technologie |
|---|---|
| Framework | Custom, Haystack 2.x als Basis-Abstraktion |
| LLM (Antworten) | `deepseek-v4-pro` via DeepSeek API (OpenAI-kompatibler Client) |
| LLM (Extraktion) | `deepseek-v4-flash` (schneller/günstiger, nur beim Indexieren) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384d, lokal |
| Vektor- & Graph-DB | Neo4j 5.x — kombinierter Vektor-Store + Knowledge Graph |
| Retrieval | Hybrid: Vektor-Suche + Graph-Traversal (2–3 Hops) + BM25 + Cross-Encoder Reranker |
| Reranker | `BAAI/bge-reranker-base` |
| Web | FastAPI + Uvicorn |
| OCR | Tesseract (`deu+eng`) für PDFs ohne Text-Layer |

### Verzeichnisstruktur

```
src/kg_rag/
├── components/       # entity_extractor, graph_retriever, bm25_retriever, reranker, context_merger
├── pipelines/        # indexing.py, query.py
├── web/              # FastAPI app (app.py)
├── config.py         # RagConfig, LLMConfig, Neo4jConfig (alle Werte via Env-Vars überschreibbar)
├── neo4j_store.py    # Neo4j-Operationen: Vektor-Suche, Graph-Traversal, CRUD
├── llm.py            # LLM-Client-Wrapper (create_chat_generator, run_chat)
└── schema.py         # Gemeinsame Datenmodelle

tests/                # Flat — keine Unterordner, alle Dateien test_*.py
```

### Eval-Ziele

Eval-Library: **Ragas**

Primäre Metriken:

- `faithfulness` — Antwort nur aus Kontext, keine Halluzinationen
- `answer_relevancy` — Antwort adressiert die gestellte Frage
- `context_precision` — Ranking der relevanten Chunks im abgerufenen Kontext
- `context_recall` — Vollständigkeit des abgerufenen Kontexts

Zielbereich: mehrsprachig Deutsch/Englisch, dokumentenbasierte Q&A.

### Konventionen

- **Tests:** pytest, flache `tests/`-Struktur, Dateinamen `test_<modul>.py`
- **Eval:** Ragas (noch nicht in `pyproject.toml` — bei Bedarf ergänzen)
- **Linting:** ruff, `line-length = 100`, Regeln: `E, F, I, UP, B`
- **Python:** `>=3.10`, `from __future__ import annotations` in allen Modulen
- **Konfiguration:** Alle Laufzeit-Parameter über Env-Vars, Defaults in `RagConfig.from_env()`

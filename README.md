# Knowledge Graph RAG System

Ein selbst gehostetes Retrieval-Augmented Generation System, das PDF- und Textdokumente in einen Neo4j-Wissensgraphen indexiert und Fragen mittels hybridem Retrieval beantwortet — Vektorähnlichkeit über Chunks kombiniert mit mehrstufiger Graphtraversierung über extrahierte Entitäten.

---

## Inhaltsverzeichnis

1. [Was das System leistet](#was-das-system-leistet)
2. [Architektur](#architektur)
3. [Graphschema](#graphschema)
4. [Datenfluss](#datenfluss)
   - [Indexierung](#indexierung)
   - [Abfrage](#abfrage)
5. [Projektstruktur](#projektstruktur)
6. [Setup](#setup)
7. [Verwendung](#verwendung)
   - [CLI](#cli)
   - [Web-UI](#web-ui)
8. [API-Referenz](#api-referenz)
9. [Umgebungsvariablen](#umgebungsvariablen)
10. [Tests](#tests)
11. [Unterschied zu einem einfachen RAG](#unterschied-zu-einem-einfachen-rag)
12. [Bekannte Einschränkungen](#bekannte-einschränkungen)
13. [Operations-Cheatsheet](#operations-cheatsheet)

---

## Was das System leistet

Gegeben einen Dokumentkorpus (PDFs, Textdateien, Markdown):

1. **Indexiert** — Dokumente werden seitenweise eingelesen, gesäubert, in satzbasierte Chunks aufgeteilt, eingebettet und pro Chunk werden typisierte Entitäten und Relationen via LLM extrahiert. Eingebettete Bilder in PDFs werden per OCR (Tesseract) erfasst.
2. **Speichert** — Chunks (mit Embeddings), Entitäten und Relationen werden in Neo4j als Property-Graph mit nativem Vektorindex persistiert. Alle Daten sind sitzungsgebunden und können nach der Sitzung automatisch gelöscht werden.
3. **Retrieves** — Fragen werden durch Kombination von Vektorsuche über Chunks und Graphtraversierung über die Entitäten beantwortet, die die Frage und die Top-Chunks erwähnen.
4. **Generiert** — Der zusammengeführte Kontext wird an ein LLM übergeben, das eine Antwort liefert, die jeden verwendeten Chunk via `[S1]`-Tags zitiert. Die Antwortgenerierung kann als Server-Sent-Event-Stream (SSE) übertragen werden.

Die Web-UI ermöglicht Drag-and-Drop von PDFs und zeigt den wachsenden Wissensgraphen in Echtzeit.

---

## Architektur

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Web UI (FastAPI + Uvicorn)                            │
│  GET  /                      → static/index.html (vis-network SPA)           │
│  GET  /api/graph             → Neo4jGraphStore.fetch_entity_graph()          │
│  POST /api/upload            → IndexingPipeline.run()  [async, 202]          │
│  GET  /api/jobs/{job_id}     → JobState (polling, 700ms-Intervall im FE)     │
│  POST /api/query             → QueryPipeline.run()                           │
│  POST /api/query/stream      → QueryPipeline.retrieve() + SSE-Token-Stream   │
│  GET  /api/document/{id}/text→ PDF-Text + OCR per Seite                      │
│  DELETE /api/document/{id}   → store.delete_document()                       │
│  POST /api/session/end       → store.delete_session()                        │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
           ┌─────────────────────────┼──────────────────────┐
           ▼                         ▼                       ▼
  ┌─────────────────┐     ┌──────────────────┐    ┌──────────────────┐
  │ IndexingPipeline│     │  QueryPipeline   │    │   CLI (kg-rag)   │
  │  load → split → │     │  embed question →│    │  setup-schema /  │
  │  embed → extract│     │  vector + section│    │  index / query / │
  │  → persist      │     │  + graph + merge │    │  serve           │
  │                 │     │  → generate      │    │                  │
  └────────┬────────┘     └────────┬─────────┘    └────────┬─────────┘
           │                       │                       │
           └───────────────┬───────┴───────────────────────┘
                           ▼
         ┌─────────────────────────────────┐
         │         Neo4jGraphStore         │
         │  - schema setup (idempotent)    │
         │  - persist (batched UNWIND)     │
         │  - vector search               │
         │  - section_title_search        │
         │  - graph traversal             │
         │  - entity_context              │
         │  - fetch_entity_graph          │
         │  - delete_document / _session  │
         │  - orphan chunk purge          │
         └────────────────┬────────────────┘
                          │ Bolt (7687)
                          ▼
         ┌─────────────────────────────────┐
         │   Neo4j 5+ (Property Graph)     │
         │   - Native Vektorindex          │
         │   - Sitzungsisolierung          │
         └─────────────────────────────────┘

Embedding-Schicht (lokal, CPU/MPS):
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  384 Dim · cosine · unterstützt Deutsch + Englisch

LLM-Schicht (Remote, OpenAI-kompatibel):
  Antwortgenerierung:  deepseek-v4-pro   (LLM_MODEL)
  Entitätsextraktion: deepseek-v4-flash  (LLM_EXTRACTION_MODEL)
  Default-Endpoint:   https://api.deepseek.com
  Beliebiger OpenAI-kompatibler Endpoint via LLM_BASE_URL austauschbar
```

### Stack

| Schicht     | Technologie                                                                                                                                                                                                                      |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM         | DeepSeek API (OpenAI-kompatibel, `https://api.deepseek.com`). Standardmodell für Antworten: `deepseek-v4-pro`, für Extraktion: `deepseek-v4-flash`. Jeder OpenAI-kompatible Endpoint kann über `LLM_BASE_URL` eingesetzt werden. |
| Embeddings  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 Dim, cosine, mehrsprachig: Deutsch + Englisch out-of-the-box)                                                                                                 |
| Pipeline    | Haystack 2.x (`OpenAIChatGenerator` gegen DeepSeek API, `SentenceTransformersDocumentEmbedder`, `DocumentSplitter`, etc.)                                                                                                        |
| OCR         | Tesseract (`pytesseract` + `pypdf` für Bildextraktion), Sprache konfigurierbar (`deu+eng`)                                                                                                                                       |
| Graph-DB    | Neo4j 5+                                                                                                                                                                                                                         |
| Web-Backend | FastAPI + Uvicorn                                                                                                                                                                                                                |
| Frontend    | Vanilla JS + `vis-network` 9.x (CDN)                                                                                                                                                                                             |
| Sprache     | Python 3.10+                                                                                                                                                                                                                     |

---

## Graphschema

```
(:Document {id, title, source, created_at, updated_at, session_id})
   -[:HAS_CHUNK]→
(:Chunk {id, text, embedding[384], chunk_index, document_id, source, title,
         page_number, section_title?, session_id})
   -[:NEXT_CHUNK]→ (:Chunk)
   -[:MENTIONS]→
(:Entity {name, name_normalized, type, description, session_id})
   -[:RELATES_TO {relation, chunk_id}]→ (:Entity)

(:IndexingMeta {session_id, model, dimensions, updated_at})
```

**Eindeutigkeitsbeschränkungen**

- `Document.id`
- `Chunk.id`
- `Entity.name_normalized` + `session_id` (Entitäten sind sitzungsisoliert)

**Vektorindex**

- `chunk_embeddings` auf `Chunk.embedding`, 384 Dimensionen, cosine-Ähnlichkeit.

**Sitzungsisolierung**  
Alle Knoten tragen `session_id`. Abfragen, Graph-Dumps und Löschoperationen werden immer auf die aktuelle Sitzung beschränkt. `POST /api/session/end` bereinigt alle Daten einer Sitzung aus Neo4j. Veraltete Chunks (nach erneutem Indexieren derselben Datei) werden nach jedem Index-Job automatisch bereinigt.

`name_normalized` ist die casefolded Version der Entitätsoberfläche. Es ist der einzige Schlüssel für Entitätsidentität — Aliase werden nicht verfolgt.

---

## Datenfluss

### Indexierung

Ausgelöst über `kg-rag index <Pfade>` oder `POST /api/upload`.

**Schritt 1 — Laden**  
Haystack-Konverter (`TextFileToDocument`, `PyPDFToDocument`) lesen `.txt`, `.md`, `.pdf`. Fallback-Loader (direkte `pypdf`-Aufrufe) werden verwendet, wenn Haystack-Konverter nicht importiert werden können. PDFs werden seitenweise aufgeteilt (`\x0c` als Seitentrennzeichen) und Inhaltsverzeichnisseiten werden automatisch erkannt und übersprungen.

**Abschnittserkennung (Section Detection)**  
Innerhalb jeder Seite erkennt ein Regex-basierter Parser Überschriften (römische Ziffern `VII. Conclusion`, Dezimalnummern `5.2 Methoden`, Schlüsselwörter auf Englisch und Deutsch wie `Abstract`, `Zusammenfassung`, `Einleitung`, `Fazit` usw.). Jeder Abschnitt wird mit seinem Titel als `section_title`-Metadatum gespeichert. Das erste Chunk jedes Abschnitts bekommt den Titel in seinen Text eingebettet — das gibt überschriften-gerichteten Abfragen einen mehrsprachigen Anker, ohne jeden Chunk derselben Überschrift zu überfrachten.

**OCR-Extraktion**  
Wenn `OCR_ENABLED=true` (Standard), werden Bilder aus jeder PDF-Seite via `pypdf` extrahiert und mit Tesseract in Text umgewandelt. Sprache: konfigurierbar via `OCR_LANGUAGE` (Standard: `deu+eng`). OCR-Text wird an den pypdf-Textinhalt angehängt. Wenn Tesseract nicht installiert ist, läuft das System ohne OCR weiter.

**Schritt 2 — Säubern & Aufteilen**  
`DocumentCleaner` + `DocumentSplitter` erzeugen satzbasierte Chunks mit konfigurierbarer Länge und Überschneidung (`CHUNK_SPLIT_LENGTH` / `CHUNK_SPLIT_OVERLAP`, Standard 10/2). Ein Regex-basierter Fallback-Splitter (mit Schutz vor Abkürzungen wie `Dr.`, `Nr.`, Dezimalzahlen) übernimmt, falls der Haystack-Splitter nicht verfügbar ist.

**Schritt 3 — Einbetten**  
`SentenceTransformersDocumentEmbedder` bettet jeden Chunk in einen 384-Dim-Vektor ein. Batching: konfigurierbar via `EMBEDDING_BATCH_SIZE` (Standard 64). Gerät: `EMBEDDING_DEVICE` (`cpu`, `mps`, `auto` — `auto` erkennt Apple Silicon MPS automatisch). Der Embedder wird beim Start gecacht (`lru_cache`), sodass nur einmal geladen wird.

**Schritt 4 — Entitätsextraktion**  
Für jeden Chunk wird das LLM mit einem strengen JSON-only Prompt aufgerufen, der `entities` (typisiert) und `relations` (Quelle → Ziel mit verb-artigem Label) verlangt. Aufrufe laufen parallel in einem gebundenen Thread-Pool (`EXTRACTION_CONCURRENCY`, Standard 30). Malformed JSON liefert eine leere Extraktion zurück. Relationen, deren Quelle/Ziel nicht in der Entitätsliste des Chunks stehen, werden verworfen. Retries: konfigurierbar via `EXTRACTION_MAX_RETRIES`. Timeout pro Chunk: `EXTRACTION_TIMEOUT_SECONDS`.

**Schritt 5 — Persistieren**  
Vier batched `UNWIND`-Cypher-Schreibvorgänge pro Indexierungslauf (nicht pro Chunk):

- **Chunks-Batch**: `MERGE` Document + Chunk + `HAS_CHUNK` für ~100 Chunks auf einmal.
- **Entitäten-Batch**: `MERGE` jede `Entity` nach `(name_normalized, session_id)` und erstellt `MENTIONS`-Kanten in einem Durchgang.
- **Relationen-Batch**: `MERGE` `Entity-[:RELATES_TO {relation, chunk_id}]->Entity`.
- **Chunk-Sequenz-Batch**: `MERGE` benachbarte `Chunk-[:NEXT_CHUNK]->Chunk`-Paare in `chunk_index`-Reihenfolge.

**Schritt 6 — Bereinigung**  
Nach der Persistierung werden veraltete Chunks (von einem früheren Indexierungslauf derselben Datei) per `delete_stale_chunks_bulk` aus Neo4j entfernt. Orphan-Chunks (Chunks ohne zugehöriges Dokument im Manifest) werden via `_purge_orphan_chunks` bereinigt.

**Schritt 7 — Metadaten speichern**  
Das verwendete Embedding-Modell und die Dimensionszahl werden als `IndexingMeta`-Knoten in Neo4j gespeichert. Die Query-Pipeline prüft bei jeder Abfrage, ob das konfigurierte Modell mit dem indexierten übereinstimmt, und wirft einen klaren Fehler mit Re-Indexierungs-Anweisung.

---

### Abfrage

Ausgelöst via `kg-rag query <Frage>`, `POST /api/query` oder `POST /api/query/stream`.

**Schritt 1 — Fragen-Entitäten & Fragen-Embedding (parallel)**  
Gleichzeitig laufen zwei Tasks:

- Das LLM extrahiert Entitätsnamen aus der Frage (JSON, gleiche Form wie beim Indexieren). Timeout: 15 s, 1 Retry.
- `SentenceTransformersTextEmbedder` erzeugt einen 384-Dim-Query-Vektor (gecacht via `lru_cache`).

Außerdem wird die Frage auf **Abschnitts-Trigger** geprüft: Enthält sie Schlüsselwörter wie `Schlussbetrachtung`, `Fazit`, `Abstract`, `Einleitung`, `Methodik`, `Ergebnisse`, `Diskussion`, `Referenzen`, `Anhang` (Englisch + Deutsch), wird parallel eine strukturelle `section_title_search` in Neo4j ausgelöst, die Chunks per Titeltreffer abruft — ohne Cosine-Schwelle.

**Schritt 2 — Vektorsuche**  
Neo4j gibt die Top-k Chunks via `db.index.vector.queryNodes` zurück. Chunks unterhalb der Ähnlichkeitsschwelle (`MIN_SIMILARITY`, Standard 0,25) werden gefiltert. Abschnitts-gematchte Docs überspringen diese Schwelle.

**Schritt 3 — Graphsuche**  
Von den Chunk-IDs der Vektorsuche und den normalisierten Fragen-Entitätsnamen wird traversiert:  
`Chunk-[:MENTIONS]->Entity-[:RELATES_TO*1..h]->Entity<-[:MENTIONS]-Chunk`  
bis zu `h` Hops (Standard 2, begrenzt auf 1–3). Seed-Chunks werden ausgeschlossen. Ergebnisse werden auf `GRAPH_LIMIT` begrenzt.

**Schritt 4 — Entitätskontext**  
Direkte `RELATES_TO`-Nachbarn der Fragen-Entitäten werden als lesbare Zeilen gesammelt:  
`Quelle --Relation--> Ziel`

**Schritt 5 — Merge**  
Vektor-Chunks, Graph-Chunks und der Entitätskontext-Block werden unter einem `MAX_CONTEXT_CHARS`-Budget zusammengeführt. Deduplizierung nach `chunk_id`. Vektor-Chunks werden zuerst eingeordnet. Jedem Chunk-Abschnitt wird ein `[S1]`-Tag vorangestellt.

Wenn weder Vektor-Chunks noch Graph-Chunks noch Entitätskontext gefunden werden, gibt das System sofort `"Der bereitgestellte Kontext reicht nicht aus, um diese Frage zu beantworten."` zurück — ohne LLM-Aufruf.

**Schritt 6 — Generieren**  
Das LLM wird mit dem zusammengeführten Kontext und der Frage aufgerufen. Der System-Prompt weist das Modell an, **ausschließlich** auf Basis des Kontexts zu antworten, keine Fakten zu erfinden, und jede genutzte Quelle via `[S1]`-Tags zu zitieren. Temperatur: 0,2. Ungültige Zitations-Tags werden nach der Generierung sanitiert.

**Streaming (`/api/query/stream`)**  
Retrieval (Schritte 1–5) läuft blockierend in einem Thread-Pool. Sobald der Kontext fertig ist, werden die LLM-Token als SSE-Events übertragen:

- `event: meta` — Metadaten (query_entities, vector_chunks, graph_chunks, citations)
- `event: token` — Jeder Token-Delta
- `event: final` — Vollständige, sanitierte Antwort
- `event: error` — Fehlermeldung

---

## Projektstruktur

```
src/kg_rag/
├── __init__.py
├── cli.py                        # argparse-Einstiegspunkt: setup-schema, index, query, serve
├── config.py                     # RagConfig + LLMConfig + Neo4jConfig (alle via Env-Vars)
├── compat.py                     # Haystack-Versions-Shims (Document, @component, …)
├── llm.py                        # create_chat_generator, run_chat, stream_chat_tokens
│                                 #   → Haystack OpenAIChatGenerator, zeigt auf DeepSeek API
├── logging.py                    # loguru-Setup + log_timing-Kontextmanager
├── neo4j_store.py                # Neo4jGraphStore:
│                                 #   setup_schema, persist_documents, vector_search,
│                                 #   section_title_search, graph_search, entity_context,
│                                 #   fetch_entity_graph, delete_document, delete_session,
│                                 #   delete_orphan_chunks, delete_stale_chunks_bulk,
│                                 #   store_indexing_meta, get_indexing_meta
├── schema.py                     # Entity / Relation / ExtractionResult Dataklassen
│                                 #   + normalize_entity_name (casefold)
├── components/
│   ├── entity_extractor.py       # @component — LLM-basierte JSON-Extraktion mit Validierung
│   │                             #   Concurrent ThreadPoolExecutor, Retry-Logik
│   ├── graph_retriever.py        # @component — graph_search + entity_context Wrapper
│   └── context_merger.py         # @component — Deduplizierung + Zeichenbudget-Merge
│                                 #   + Citation-Index-Vergabe ([S1], [S2], …)
├── pipelines/
│   ├── indexing.py               # IndexingPipeline + Hilfsfunktionen:
│   │                             #   collect_supported_files, load_documents,
│   │                             #   split_documents, embed_documents,
│   │                             #   normalize_chunk_metadata, fallback_sentence_split,
│   │                             #   _segment_into_sections (Abschnittserkennung),
│   │                             #   _is_toc_page (ToC-Erkennung),
│   │                             #   _load_pdf_documents (ProcessPoolExecutor für mehrere PDFs),
│   │                             #   _ocr_pages (Tesseract-Integration)
│   └── query.py                  # QueryPipeline + QueryResult + RetrievalResult
│                                 #   + embed_query, sanitize_citations,
│                                 #   _extract_section_keywords, invalidate_embedding_meta
└── web/
    ├── __init__.py
    ├── app.py                    # FastAPI-Factory + uvicorn-Runner
    │                             #   JobState, FileEntry (Async-Upload mit Fortschritt)
    │                             #   PDF-Seitenextraktion mit In-Memory-Cache
    │                             #   Orphan-Chunk-Bereinigung nach jedem Job
    │                             #   Embedder-Warmup beim Start
    └── static/
        └── index.html            # Drag-Drop + vis-network Visualisierung
                                  #   Two-Tab UI: Graph + Chat
                                  #   SSE-Streaming-Antworten mit Zitations-Chips

tests/                            # Alle Tests mit Fakes — kein Neo4j, kein Netz, kein LLM-Key
docs/                             # Beispiel-Markdown zum Indexieren
docker-compose.yml                # Optionaler Docker-basierter Neo4j-Dienst
pyproject.toml
.env.example
```

---

## Setup

### 1. Neo4j installieren und starten

Der einfachste Weg unter macOS via Homebrew (kein Desktop-GUI nötig):

```bash
brew install neo4j
neo4j-admin dbms set-initial-password password123
brew services start neo4j
```

Überprüfung: http://localhost:7474 — Neo4j Browser sollte laden.

Starten/Stoppen: `brew services stop neo4j` / `brew services start neo4j`

**Alternativen:**

- Neo4j Desktop: https://neo4j.com/download/
- Community ZIP von neo4j.com
- Docker: `docker compose up -d neo4j` (siehe `docker-compose.yml`)

---

### 2. OCR-Unterstützung (optional, aber empfohlen)

Um Text aus bild-basierten PDFs und Scans zu extrahieren:

```bash
# macOS
brew install tesseract tesseract-lang

# Überprüfung
tesseract --version
```

Ohne Tesseract läuft das System problemlos weiter — OCR-Extraktion wird für diese Sitzung übersprungen. Konfigurierbar via `OCR_ENABLED=false`, um OCR komplett zu deaktivieren.

---

### 3. DeepSeek API-Key

1. API-Key erstellen auf https://platform.deepseek.com/api_keys
2. Guthaben aufladen auf demselben Dashboard (Pay-as-you-go, Anfragen werden bei Guthaben = 0 abgelehnt)
3. Standardmodell: `deepseek-v4-pro` (Antwortgenerierung), `deepseek-v4-flash` (Entitätsextraktion — schneller und günstiger)
4. Via `LLM_MODEL` / `LLM_EXTRACTION_MODEL` wechselbar. `LLM_BASE_URL` erlaubt jeden anderen OpenAI-kompatiblen Endpoint.

---

### 4. Python-Umgebung

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dies installiert: Haystack 2.x, OpenAI-SDK (gegen DeepSeek API gerichtet), Neo4j-Treiber, sentence-transformers (zieht PyTorch), pypdf, pytesseract, Pillow, FastAPI/Uvicorn, python-multipart sowie das Test-Toolchain (pytest, httpx, ruff).

---

### 5. Konfigurieren

```bash
cp .env.example .env
```

`.env` bearbeiten. Der einzige Pflichteinttrag ist `LLM_API_KEY`. Alle anderen Variablen haben sinnvolle Standardwerte.

---

### 6. Neo4j-Schema erstellen

```bash
kg-rag setup-schema
```

Legt drei Eindeutigkeitsbeschränkungen und den Vektorindex an. **Idempotent** — kann jederzeit ohne Datenverlust erneut ausgeführt werden.

---

## Verwendung

### CLI

```bash
# Einmalig pro Neo4j-Instanz
kg-rag setup-schema

# Verzeichnis rekursiv indexieren
kg-rag index ./docs

# Graph vor dem Indexieren leeren (MATCH (n) DETACH DELETE n → setup-schema → index)
kg-rag index ./paper.pdf --overwrite

# Frage stellen (Deutsch oder Englisch)
kg-rag query "Wie hängt X mit Y zusammen?"

# Mit erweiterter Ausgabe
kg-rag query "..." --top-k 5 --hops 2 --show-context

# Web-UI starten
kg-rag serve --host 127.0.0.1 --port 8000
```

`--overwrite` führt `MATCH (n) DETACH DELETE n` gefolgt von `setup-schema` aus, bevor indexiert wird. Ohne `--overwrite` werden neue Dokumente in den bestehenden Graphen gemergt.

---

### Web-UI

```bash
kg-rag serve
```

Öffne http://127.0.0.1:8000/. Die UI hat zwei Tabs:

**Tab „Graph"**  
Rendert alle Entitäten und `RELATES_TO`-Kanten via `vis-network`. Wird nach jedem erfolgreichen Upload automatisch aktualisiert. Knoten sind nach Entitätstyp eingefärbt; Kanten zeigen die Relationsbezeichnung.

**Tab „Chat"**  
Einzel-Turn Q&A gegen das indexierte Korpus. Antworten werden als SSE-Stream Token für Token geliefert. Jede Antwort zeigt:

- Zitations-Chips (`[S1]`, `[S2]` usw.) — klickbar, um den Quell-Chunk zu öffnen
- Anzahl der Vektor-Chunks und Graph-Chunks, die den Kontext gespeist haben
- Die Entitäten, die das LLM aus der Frage extrahiert hat

**Sidebar (Upload)**  
Drag-and-Drop für PDFs (einzeln oder mehrere gleichzeitig, max. 50 MB pro Datei, ausschließlich PDF). Zeigt für jede Datei einen Live-Fortschrittsbalken mit den Phasen `parsing → splitting → embedding → extracting → persisting → done`. Fehler werden pro Datei und als globaler Status angezeigt.

**Dokument-Viewer**  
Klick auf einen Zitations-Chip öffnet das Quelldokument mit dem markierten Chunk — pypdf-Text und OCR-Inhalt werden seitenweise angezeigt.

---

## API-Referenz

### `GET /`

Gibt die SPA (`index.html`) zurück.

---

### `GET /api/graph`

Gibt alle Entitäten und Relationen der aktuellen Sitzung zurück.

**Header:** `X-Session-Id: <session-id>` (Pflicht)

**Antwort:**

```json
{
  "nodes": [{ "id": "...", "label": "Entitätsname", "type": "Konzept" }],
  "edges": [{ "from": "...", "to": "...", "label": "RELATES_TO-Bezeichnung" }]
}
```

---

### `POST /api/upload`

Lädt eine oder mehrere PDFs hoch und startet einen Indexierungsjob im Hintergrund.

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Body:** `multipart/form-data`, Feld `files` (ein oder mehrere `.pdf`)  
**Max. Dateigröße:** 50 MB pro Datei  
**Status:** `202 Accepted`

**Antwort:**

```json
{
  "job_id": "a1b2c3...",
  "files": [{ "filename": "paper.pdf", "document_id": "abc123..." }],
  "status": "queued",
  "estimated_seconds": 45
}
```

Der Schätzwert basiert auf der Dateigröße: `30s Basis + 4s/MB`.

---

### `GET /api/jobs/{job_id}`

Gibt den aktuellen Status eines Indexierungsjobs zurück.

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Fehler:** `404` wenn Job nicht gefunden oder zu einer anderen Sitzung gehört.

**Antwort (JobState):**

```json
{
  "job_id": "a1b2c3...",
  "files": [
    {
      "filename": "paper.pdf",
      "document_id": "abc123...",
      "status": "done",
      "error": null
    }
  ],
  "status": "running",
  "step": "extracting",
  "current": 42,
  "total": 120,
  "chunks_indexed": 0,
  "error": null,
  "estimated_seconds": 45,
  "started_at": 1716300000.0,
  "finished_at": null,
  "graph": null
}
```

`step` durchläuft: `queued → parsing → splitting → embedding → extracting → persisting → done`  
`status` durchläuft: `queued → running → done | error`  
Wenn `status == "done"`, enthält `graph` die aktualisierten Graph-Daten.  
Jobs werden nach 1 Stunde automatisch aus dem In-Memory-Store gelöscht.

---

### `POST /api/query`

Führt eine vollständige RAG-Abfrage durch (synchron).

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Body:**

```json
{
  "question": "Wie funktioniert X?",
  "top_k": 5,
  "hops": 2
}
```

`top_k`: 1–50 (optional, Standard: `QUERY_TOP_K`)  
`hops`: 1–3 (optional, Standard: `GRAPH_HOPS`)

**Antwort:**

```json
{
  "answer": "X funktioniert durch... [S1] [S3]",
  "query_entities": ["X", "Y"],
  "vector_chunks": 5,
  "graph_chunks": 3,
  "context": "--- [S1] paper.pdf (Chunk 4, Seite 2) ---\n...",
  "citations": [
    { "index": 1, "chunk_id": "abc...", "title": "paper.pdf", "page_number": 2 }
  ]
}
```

---

### `POST /api/query/stream`

Wie `/api/query`, aber die Antwort wird als **Server-Sent Events (SSE)** gestreamt.

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Body:** Identisch mit `/api/query`  
**Content-Type:** `text/event-stream`

**Event-Sequenz:**

```
event: meta
data: {"query_entities": [...], "vector_chunks": 5, "graph_chunks": 3, "citations": [...]}

event: token
data: {"delta": "X "}

event: token
data: {"delta": "funktioniert "}

... (ein Event pro Token)

event: final
data: {"answer": "X funktioniert durch... [S1]"}
```

Bei Fehler: `event: error` mit `{"detail": "..."}`.

---

### `GET /api/document/{document_id}/text`

Gibt den extrahierten Text eines Dokuments seitenweise zurück (pypdf + OCR).

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Antwort:**

```json
{
  "document_id": "abc123...",
  "title": "paper.pdf",
  "pages": [
    {
      "page_number": 1,
      "text": "Einleitung\n\nDiese Arbeit beschäftigt sich mit..."
    },
    { "page_number": 2, "text": "..." }
  ]
}
```

Ergebnisse werden in einem In-Memory-Cache gehalten (bis Dokument gelöscht oder Sitzung beendet wird). Nach dem Upload werden PDF-Seiten proaktiv im Hintergrund vorgeladen.

---

### `DELETE /api/document/{document_id}`

Löscht ein einzelnes Dokument und alle zugehörigen Chunks, Entitäten und Relationen aus Neo4j.

**Header:** `X-Session-Id: <session-id>` (Pflicht)  
**Antwort:** `{"deleted": "<document_id>"}`

---

### `POST /api/session/end`

Bereinigt alle Sitzungsdaten aus Neo4j und löscht hochgeladene Dateien vom Dateisystem.

**Body:** `{"session_id": "<session-id>"}`  
**Antwort:** `{"deleted": "<session_id>"}`

---

## Umgebungsvariablen

Alle Variablen können in `.env` gesetzt werden (wird beim Start geladen).

### LLM

| Variable               | Standard                   | Beschreibung                                                                                                           |
| ---------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `LLM_API_KEY`          | — **(Pflicht)**            | DeepSeek API-Key (oder ein beliebiger OpenAI-kompatibler Key wenn `LLM_BASE_URL` gesetzt ist)                          |
| `LLM_MODEL`            | `deepseek-v4-pro`          | Modell für Antwortgenerierung und query-seitige Entitätsextraktion                                                     |
| `LLM_EXTRACTION_MODEL` | `deepseek-v4-flash`        | Modell für die per-Chunk-Entitätsextraktion beim Indexieren. Standardmäßig ein günstigeres/schnelleres Schwestermodell |
| `LLM_BASE_URL`         | `https://api.deepseek.com` | OpenAI-kompatibler Base-URL. Überschreiben, um einen anderen Provider zu nutzen (z.B. OpenAI, Groq, lokales Ollama)    |

### Neo4j

| Variable         | Standard                | Beschreibung                                                                            |
| ---------------- | ----------------------- | --------------------------------------------------------------------------------------- |
| `NEO4J_URI`      | `bolt://localhost:7687` | Bolt-Verbindungs-URI                                                                    |
| `NEO4J_USERNAME` | `neo4j`                 | Datenbankbenutzername                                                                   |
| `NEO4J_PASSWORD` | `password123`           | Muss mit dem beim `neo4j-admin dbms set-initial-password` gesetzten Wert übereinstimmen |
| `NEO4J_DATABASE` | `neo4j`                 | Datenbankname                                                                           |

### Embeddings

| Variable               | Standard                                                      | Beschreibung                                                                                                                     |
| ---------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `EMBEDDING_MODEL`      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace-Modell-ID. Muss 384-Dim-Vektoren erzeugen, um mit dem Vektorindex übereinzustimmen. Mehrsprachig: Deutsch + Englisch |
| `EMBEDDING_DIMENSIONS` | `384`                                                         | Dimensionszahl des Vektorindex. Bei Wechsel des Embedding-Modells anpassen und alle Dokumente neu indexieren                     |
| `EMBEDDING_BATCH_SIZE` | `64`                                                          | Anzahl Chunks pro Embedding-Batch. Erhöhen bei GPUs mit viel VRAM                                                                |
| `EMBEDDING_DEVICE`     | `cpu`                                                         | `cpu`, `mps` (Apple Silicon), `cuda`, oder `auto` (erkennt MPS automatisch)                                                      |

### Indexierung

| Variable                     | Standard  | Beschreibung                                                                                                              |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------- |
| `CHUNK_SPLIT_LENGTH`         | `10`      | Sätze pro Chunk                                                                                                           |
| `CHUNK_SPLIT_OVERLAP`        | `2`       | Satz-Überschneidung zwischen Chunks                                                                                       |
| `EXTRACTION_CONCURRENCY`     | `30`      | Parallele LLM-Aufrufe während der Entitätsextraktion. Erhöhen für schnelleres Indexieren, wenn der Provider dies verträgt |
| `EXTRACTION_TIMEOUT_SECONDS` | `600`     | Gesamtes Timeout für den Entitätsextraktionsschritt eines Indexierungsjobs                                                |
| `EXTRACTION_MAX_RETRIES`     | `4`       | Anzahl der Wiederholungsversuche bei fehlgeschlagener Entitätsextraktion pro Chunk                                        |
| `ENTITY_MAX_TOKENS`          | `1200`    | Max. LLM-Output-Tokens für die Entitätsextraktion pro Chunk                                                               |
| `OCR_ENABLED`                | `true`    | OCR für bild-basierte PDF-Seiten aktivieren. Erfordert Tesseract                                                          |
| `OCR_LANGUAGE`               | `deu+eng` | Tesseract-Sprache(n). Mehrere via `+` verbinden                                                                           |

### Abfrage

| Variable                 | Standard | Beschreibung                                                                   |
| ------------------------ | -------- | ------------------------------------------------------------------------------ |
| `QUERY_TOP_K`            | `20`     | Vektor-Treffer vor der Graph-Erweiterung                                       |
| `GRAPH_HOPS`             | `2`      | Graphtraversierungs-Tiefe. Wird auf 1–3 begrenzt                               |
| `GRAPH_MAX_HOPS`         | `3`      | Absolute Obergrenze für Hops                                                   |
| `GRAPH_LIMIT`            | `8`      | Max. Graph-Chunks pro Abfrage                                                  |
| `MIN_SIMILARITY`         | `0.25`   | Cosine-Ähnlichkeitsschwelle. Chunks unterhalb dieser Schwelle werden verworfen |
| `MAX_CONTEXT_CHARS`      | `16000`  | Zeichenbudget für den zusammengeführten Kontextblock                           |
| `ANSWER_MAX_TOKENS`      | `1500`   | Max. LLM-Output-Tokens für die Antwortgenerierung                              |
| `ANSWER_TIMEOUT_SECONDS` | `60`     | Hartes Timeout (Sekunden) für den Antwort-LLM-Aufruf                           |
| `ANSWER_MAX_RETRIES`     | `2`      | Wiederholungsversuche bei fehlgeschlagener Antwortgenerierung                  |

---

## Tests

```bash
pytest -q
```

Alle Tests verwenden Fakes — kein Neo4j, kein Netz, kein LLM-Key nötig.

| Testdatei                   | Abdeckung                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `test_indexing_helpers.py`  | Lokales Parsen, Chunk-Metadaten-Normalisierung, Abschnittserkennung, ToC-Erkennung                                       |
| `test_indexing_sections.py` | Satzbasiertes Splitting, Fallback-Splitter, Abschnittstitel-Einbettung                                                   |
| `test_context_merger.py`    | Deduplizierung von Chunks, Zeichenbudget, Citation-Index-Vergabe                                                         |
| `test_entity_extractor.py`  | JSON-Parsing der Extraktion, Drop malformed relations, Parallelverarbeitung                                              |
| `test_neo4j_store.py`       | MERGE-Statements für Chunks, Entitäten, Relationen, NEXT_CHUNK; Hops-Begrenzung auf 1–3; section_title_search            |
| `test_web_app.py`           | Graph-Payload-Form, Upload-Validierung (kein nicht-PDF, max. 50 MB), HTML-Serving, Query-Endpoint mit gemockter Pipeline |

```bash
# Mit Coverage-Bericht
pytest --cov=kg_rag --cov-report=term-missing -q

# Einzelne Testdatei
pytest tests/test_entity_extractor.py -v
```

---

## Unterschied zu einem einfachen RAG

Ein einfaches RAG ruft Chunks per Vektorähnlichkeit ab und übergibt sie ans LLM. Dieses System fügt drei Zutaten hinzu:

**1. Per-Chunk-Wissensgraph-Extraktion**  
Entitäten und typisierte Relationen werden als Graphstruktur persistiert, nicht als Freitext. „Was ist mit X verbunden?" wird zu einer 1-Hop-Cypher-Abfrage.

**2. Strukturelles Abschnitts-Retrieval**  
Fragen nach spezifischen Abschnitten (`Fazit`, `Methodik`, `Abstract` usw.) triggern eine strukturelle Suche nach `section_title` — unabhängig von Cosine-Ähnlichkeit. Das bedeutet: auch wenn die Frageformulierung semantisch weit vom Chunk-Inhalt liegt, wird der richtige Abschnitt gefunden.

**3. Hybrides Retrieval**  
Top-k Chunks kommen aus der Vektorsuche; weitere Chunks werden via Graphtraversierung über die Entitäten eingebracht, die diese Chunks erwähnen und die in der Frage selbst erkannt wurden. Beide Sets werden unter einem Zeichenbudget zusammengeführt.

**Trade-offs:**

- Indexierung ist teurer — ein LLM-Aufruf pro Chunk (abgemildert durch hohes Concurrency-Default und günstigeres Extraktionsmodell).
- Retrieval-Qualität hängt von der Extraktionsqualität ab. Ein schwaches Modell erzeugt einen spärlichen, verrauschten Graphen, der schlechter abschneidet als reine Vektorsuche.
- Für Korpora, bei denen die meisten Abfragen einfache Lookups sind, fügt die Graphschicht Latenz ohne Mehrwert hinzu.

---

## Bekannte Einschränkungen

- **Entitätsidentität:** `name_normalized` (casefold only). Aliase, Pluralformen und geringfügige Schreibvarianten werden als separate Entitäten behandelt.
- **Upload-History:** Nur im Seitenkontext — setzt sich beim Reload zurück.
- **Chat:** Single-Turn — kein Gesprächsverlauf wird ans LLM zurückgegeben.
- **Kein Re-Ranking:** Zwischen Vektorergebnissen und LLM gibt es keine Re-Ranking-Stufe.
- **Embedding-Modelwechsel:** Das Wechseln des Embedding-Modells nach der Indexierung führt zu einem Fehler bei der nächsten Abfrage. Alle Dokumente müssen neu indexiert werden.
- **Einzelner Indexierungsjob:** Nur ein Indexierungsjob läuft gleichzeitig (1-Worker-ThreadPoolExecutor). Weitere Uploads werden in der Warteschlange gehalten.

---

## Operations-Cheatsheet

```bash
# Alles starten
brew services start neo4j
source .venv/bin/activate
kg-rag serve

# Alles stoppen
brew services stop neo4j
pkill -f "uvicorn.*kg_rag.web"

# Graph im Browser inspizieren
open http://localhost:7474

# Graph zurücksetzen
kg-rag index ./docs --overwrite

# Logs beobachten
kg-rag serve 2>&1 | tee kg-rag.log

# Tesseract-Verfügbarkeit prüfen
tesseract --version

# Alle Tests ausführen
pytest -q

# Linting
ruff check src/ tests/
```

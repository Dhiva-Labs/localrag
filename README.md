# LocalRAG

Fully offline, self-hosted RAG server — documents in, cited answers out. No API keys, no cloud.

[![CI](https://github.com/Dhiva-Labs/localrag/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhiva-Labs/localrag/actions/workflows/ci.yml)

## Features

- Ingest PDF, TXT, and Markdown files with per-page tracking
- Content-hash deduplication — re-uploading the same file is a no-op
- Token-aware chunking that never crosses a page boundary, with configurable overlap
- Local embeddings via sentence-transformers (`BAAI/bge-small-en-v1.5`), fully offline after the first download
- Vector storage in ChromaDB, cosine similarity
- Top-k retrieval with a score threshold — if nothing clears the bar, the server says so instead of guessing
- Streamed answers over Server-Sent Events, backed by a local Ollama model
- Page-level citations with similarity scores on every answer
- Single-file web UI: no build step, no external network requests
- CLI for headless and bulk ingestion
- Docker Compose setup for a two-container deployment (LocalRAG + Ollama)

## How it works

```
browser UI --fetch()/SSE--> FastAPI  (/api/ingest, /api/query, ...)

  ingest   extract -> chunk -> embed (bge-small) -> ChromaDB

  query    ChromaDB -> retrieve -> build prompt -> Ollama (llama3.2)
                                                        |
                                                        v
                                    streamed answer + citations
```

Ingest and query are independent paths that share the same vector store.
Ingest turns a file into page-tagged chunks and stores their embeddings.
Query embeds the question, retrieves the closest chunks, and — only if at
least one clears the score threshold — asks Ollama to answer from that
context, streaming the response back token by token.

## Quickstart

### Option A: pip

Prerequisites: Python 3.10+, and [Ollama](https://ollama.com) installed and
running with the generation model pulled once:

```sh
ollama pull llama3.2
```

Then:

```sh
git clone https://github.com/Dhiva-Labs/localrag.git
cd localrag
pip install .
localrag serve
```

Open http://localhost:8090.

The first document you ingest triggers a one-time download of the bge-small
embedding model (about 130 MB). After that, nothing LocalRAG does needs a
network connection.

### Option B: Docker

Prerequisites: Docker and Docker Compose.

```sh
git clone https://github.com/Dhiva-Labs/localrag.git
cd localrag
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2   # one-time
```

Open http://localhost:8090.

Documents, the vector store, and the downloaded embedding model live in the
`localrag-data` named volume; pulled Ollama models live in the `ollama`
named volume. Both persist across `docker compose down` (but not `down -v`).

## CLI

```
localrag serve
```

Starts the web server, configured entirely from the environment (see
Configuration below).

```
localrag ingest ./folder
```

Ingests a single file or a directory, recursively, over `*.pdf`, `*.txt`,
and `*.md`. Files already indexed (matched by content hash) are reported as
duplicates rather than re-processed. Prints one line per file plus a summary
line, and exits with status 1 if nothing ended up ingested.

## API reference

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/ingest` | multipart form, one or more `files` fields (`.pdf`/`.txt`/`.md`) | `{"documents": [{"doc_id", "filename", "pages", "chunks", "deduped"}]}` |
| GET | `/api/docs` | — | `{"documents": [{"doc_id", "filename", "sha256", "pages", "chunks", "ingested_at"}]}` |
| DELETE | `/api/docs/{doc_id}` | — | `204 No Content`, or `404` if the id is unknown |
| POST | `/api/query` | `{"question": "...", "top_k": 5}` (`top_k` optional, clamped to 1-50) | `text/event-stream`, see below |
| GET | `/api/health` | — | `{"status", "chroma", "ollama", "model", "docs"}` |
| GET | `/` | — | the web UI |

A `POST /api/ingest` request is all-or-nothing: if any file is unsupported
or empty, the whole request is rejected with `400` and a message naming the
offending file, and nothing in the request is ingested.

### `/api/query` event stream

Each answer is a sequence of Server-Sent Events: zero or more `token`
events, one `citations` event, then one `done` event. Example frame:

```
event: token
data: {"text": "Wind "}

event: token
data: {"text": "turbines "}

event: citations
data: [{"doc": "notes.txt", "page": 1, "snippet": "Wind turbines convert...", "score": 0.812}]

event: done
data: {}
```

If Ollama fails partway through a stream, an `event: error` frame with
`{"message": "..."}` is sent in place of the citations event, followed by
`done`. Any tokens already streamed stay in the answer.

**Refusal behavior.** If no stored chunk clears `LOCALRAG_SCORE_THRESHOLD`
for a question, LocalRAG never calls Ollama. It emits a single `token`
event with the fixed text `I don't know based on the indexed documents.`,
an empty `citations` array, then `done`. This is the mechanism that keeps
answers grounded — the model is never asked to answer from silence.

## Configuration

All settings are environment variables, read once at startup.

| Variable | Default | Meaning |
|---|---|---|
| `LOCALRAG_HOST` | `127.0.0.1` | interface the server binds to |
| `LOCALRAG_PORT` | `8090` | port the server listens on |
| `LOCALRAG_DATA_DIR` | `./data` | where the vector store and document registry live |
| `LOCALRAG_OLLAMA_URL` | `http://localhost:11434` | base URL of the Ollama server |
| `LOCALRAG_MODEL` | `llama3.2` | Ollama model used to generate answers |
| `LOCALRAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model used for embeddings |
| `LOCALRAG_TOP_K` | `5` | default number of chunks retrieved per query |
| `LOCALRAG_SCORE_THRESHOLD` | `0.30` | minimum similarity score a chunk must clear to be used |
| `LOCALRAG_CHUNK_TOKENS` | `500` | maximum tokens per chunk |
| `LOCALRAG_CHUNK_OVERLAP` | `50` | approximate token overlap between consecutive chunks on the same page |

## FAQ

**Can I use a different generation model?**
Set `LOCALRAG_MODEL` to any model you've pulled into Ollama, e.g.
`ollama pull mistral` then `LOCALRAG_MODEL=mistral localrag serve`.

**Where does my data live, and how do I wipe it?**
Everything — the document registry and the vector store — lives under
`LOCALRAG_DATA_DIR` (`./data` by default, or the `localrag-data` volume
under Docker). Delete that directory, or the volume, to start clean.

**Why is the first answer after startup slow?**
Ollama loads the model into memory on its first request; that cold load
can take a while depending on model size and hardware. Later requests are
fast. LocalRAG's streaming client accounts for this and won't time out
waiting for the first token.

**How does "I don't know" work — is this actually grounded?**
Every question is answered only from chunks whose similarity score clears
`LOCALRAG_SCORE_THRESHOLD`. If nothing clears it, LocalRAG refuses before
ever calling the model — see the refusal behavior section above.

**Can it read scanned PDFs?**
Not in v0.1. Extraction pulls the text layer already embedded in the PDF;
there's no OCR step for scanned or image-only pages.

**What about Word documents?**
Planned, not in v0.1. Only `.pdf`, `.txt`, and `.md` are supported today.

**Does anything ever leave my machine?**
No, with one exception: the embedding model download the first time you
ingest a document, and whatever `ollama pull` fetches for the generation
model. Both are one-time and cached locally. After that, LocalRAG makes no
outbound network calls — it's safe to run fully air-gapped.

## License

MIT

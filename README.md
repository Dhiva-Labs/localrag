# LocalRAG

Fully offline, self-hosted RAG server — documents in, cited answers out. No API keys, no cloud.

**Documentation:** step-by-step guides at [learn.dhivalabs.com](https://learn.dhivalabs.com/apps/localrag/): [install](https://learn.dhivalabs.com/apps/localrag/install) · [beginner guide](https://learn.dhivalabs.com/apps/localrag/getting-started) · [troubleshooting](https://learn.dhivalabs.com/apps/localrag/faq). New to the topic? Try the [AI and local models](https://learn.dhivalabs.com/tracks/ai/) learning track.

[![ARM64 builds](https://github.com/Dhiva-Labs/localrag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Dhiva-Labs/localrag/actions/workflows/ci.yml)
![Linux x64 | arm64](https://img.shields.io/badge/Linux-x64%20%7C%20arm64-informational?logo=linux&logoColor=white)
![Docker amd64 | arm64](https://img.shields.io/badge/Docker-amd64%20%7C%20arm64-informational?logo=docker&logoColor=white)

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

## Engineering highlights

- **Page-level citations.** Extraction keeps page numbers, and the chunker never lets a chunk cross a page boundary, so every citation names a document and page. Citations are deduplicated to the best score per document and page. See [`core/chunker.py`](localrag/core/chunker.py), [`api/query.py`](localrag/api/query.py), and the chunker tests in [`tests/test_chunker.py`](tests/test_chunker.py).
- **Refusal when retrieval is weak.** If no chunk clears `LOCALRAG_SCORE_THRESHOLD`, the server answers with a fixed "I don't know" and never calls Ollama. This is tested by [`test_query_refusal_path_never_calls_ollama`](tests/test_api.py); the filter is in [`core/retriever.py`](localrag/core/retriever.py).
- **SSE streaming with defined failure behaviour.** Tokens stream as they arrive, followed by one `citations` and one `done` event. If Ollama fails mid-stream, an `error` event is sent after the tokens already delivered ([`test_query_error_path_emits_error_then_done_after_partial_tokens`](tests/test_api.py)).
- **Idempotent ingest.** Files are hashed by content; re-uploading the same bytes is a no-op, and a request with any bad file is rejected as a whole ([`api/ingest.py`](localrag/api/ingest.py), [`tests/test_api.py`](tests/test_api.py)).
- **End-to-end check in CI.** [`ci.yml`](.github/workflows/ci.yml) builds the Docker image on Linux x64 and arm64, starts the Docker Compose stack, ingests a document, asks a question against a small Ollama model (`qwen2.5:0.5b`), and asserts the stream contains tokens, a citation to the ingested file, and `done` with no error.

## Status and limits

Version 0.1.0, a small reference build. The unit and API tests above use fakes for the embedder and Ollama; only the CI compose job exercises the real stack, and it checks that an answer is streamed and cited, not that it is correct.

- **Answer quality and retrieval accuracy are not measured.** There is no evaluation set in this repo, so no accuracy, recall or latency figures are claimed. The default score threshold (0.30) is a starting value, not a tuned one.
- No authentication on the API; bind it to localhost or put it behind your own proxy.
- PDF support reads the embedded text layer only. No OCR, no Word documents.

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

### Platforms: Linux x64 and arm64

LocalRAG runs on x86-64 and arm64 (aarch64) Linux, with Docker or with pip.

- **Docker**: `docker compose up --build` builds the image for the host's
  architecture. The CPU-only PyTorch wheels come from
  `https://download.pytorch.org/whl/cpu`, which publishes both `x86_64` and
  `aarch64` builds, and every other dependency installs from a prebuilt wheel,
  so no compiler is involved. The `ollama/ollama` image used by the compose
  file is published for both architectures too. Nothing is pushed to a
  registry; you build the image on the machine that runs it.
- **pip**: the same command works on both. To avoid pulling CUDA libraries on
  x86-64, install the CPU-only torch first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
  `pip install .`.
- **CI** runs the test suite on Ubuntu 24.04 (x64) and Ubuntu 24.04 arm64
  (Python 3.10 and 3.12), and on each of those runners builds the Docker image
  natively as a check, verifies that torch is the CPU-only build and that every
  compiled library in the image targets the runner's architecture, starts the
  compose stack, ingests a document, and gets a streamed, cited answer from a
  small Ollama model (`qwen2.5:0.5b`). The images are discarded after the run.

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

## Development

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, avoids CUDA libraries
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

These are the same lint, format and test steps CI runs (see [`ci.yml`](.github/workflows/ci.yml)).

## Sponsors

Dhiva Labs is an independent, self-funded lab. [Sponsorships](https://github.com/sponsors/Dhiva-Labs) help bring this project and the rest of Dhiva Labs to every platform, iPhone and Mac included.

[![Sponsor Dhiva Labs](https://img.shields.io/badge/Sponsor-Dhiva%20Labs-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Dhiva-Labs)

Monthly sponsors at $10 and up are listed here:

<!-- sponsors -->
*Be the first: [sponsor Dhiva Labs](https://github.com/sponsors/Dhiva-Labs).*
<!-- sponsors -->

## License

MIT

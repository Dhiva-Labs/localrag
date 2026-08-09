# Changelog

## 0.1.0 — 2026-08-10

Initial release.

- Ingest PDF, TXT, and Markdown documents with per-page tracking and
  content-hash deduplication
- Token-aware chunking (500 tokens, 50 overlap) that never crosses a page
  boundary
- Local embeddings with `BAAI/bge-small-en-v1.5`, persisted in ChromaDB
- Top-k retrieval with a similarity-score threshold and a grounded refusal
  path — below-threshold questions are answered with a fixed "I don't know"
  instead of a guess
- Streamed answers over Server-Sent Events from a local Ollama model, with
  page-level citations on every answer
- Single-file web UI: drag-drop upload, document list with delete, streaming
  chat with citation chips, first-token latency readout
- CLI: `localrag serve` and `localrag ingest <path>` for bulk ingestion
- Docker Compose deployment running LocalRAG and Ollama together

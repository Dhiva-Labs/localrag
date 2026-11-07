# Retrieval and refusal eval

A small golden set that checks two things on the real ingest and retrieval path:

- retrieval hit rate: is the expected document and page in the top k?
- refusals: questions the documents cannot answer should return nothing above the
  score threshold (correct refusal), and answerable questions should not (wrongful refusal).

Generated answer text is not scored.

- `golden.json`: 37 answerable questions (expected document and page) and 10 unanswerable
  ones, either near misses about a covered topic or completely off topic.
- `docs/`: four fictional documents (three PDFs, one Markdown file), so answers cannot come
  from general knowledge. `make_docs.py` regenerates the PDFs with only the standard library.
- `run_eval.py`: runs everything and prints a report.

```
pip install -e .
python eval/run_eval.py            # --k, --threshold, --json out.json
```

Embeddings use sentence-transformers (`LOCALRAG_EMBED_MODEL`, default `BAAI/bge-small-en-v1.5`),
not Ollama, and the refusal decision is made before any generation, so Ollama is not needed.
The first run downloads the embedding model. Results depend on the embedder; the report
prints which one was used. Results are not committed. The manual `Eval` GitHub workflow
runs the script and uploads the report.

A small author-written set like this does not predict real-world quality.

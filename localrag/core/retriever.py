"""Top-k retrieval with a score-threshold cutoff."""

from typing import Any

from localrag.core.store import RetrievedChunk, Store


def retrieve(
    question: str,
    *,
    embedder: Any,
    store: Store,
    top_k: int,
    score_threshold: float,
) -> list[RetrievedChunk]:
    query_embedding = embedder.embed_query(question)
    results = store.query(query_embedding, top_k)
    return [chunk for chunk in results if chunk.score >= score_threshold]

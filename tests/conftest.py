"""Shared test fixtures: fast deterministic fakes standing in for real models."""

import hashlib
import math

_EMBED_DIMS = 64


def _hash_bag_of_words(text: str, dims: int = _EMBED_DIMS) -> list[float]:
    vector = [0.0] * dims
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        dim = int.from_bytes(digest[:4], "big") % dims
        vector[dim] += 1.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class FakeEmbedder:
    """Hashed bag-of-words stand-in for Embedder: no model, fully deterministic."""

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [_hash_bag_of_words(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_bag_of_words(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

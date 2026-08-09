"""Sentence embedding wrapper. Keep this module import cheap and offline-safe."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# bge v1.5 models are trained asymmetrically: queries get this instruction
# prefix, indexed passages do not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([QUERY_PREFIX + text])[0]

    def count_tokens(self, text: str) -> int:
        model = self._load()
        return len(model.tokenizer.encode(text))

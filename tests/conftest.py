"""Shared test fixtures: fast deterministic fakes standing in for real models."""

import hashlib
import math
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from localrag.config import Settings
from localrag.server import create_app

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


class FakeOllama:
    """Canned stand-in for OllamaClient: no HTTP, no subprocess, always up."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["This ", "is ", "a ", "fake ", "answer."]

    async def is_up(self) -> bool:
        return True

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        for token in self.tokens:
            yield token


def make_test_settings(data_dir: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8090,
        data_dir=data_dir,
        ollama_url="http://localhost:11434",
        model="test-model",
        embed_model="test-embed-model",
        top_k=5,
        score_threshold=0.30,
        chunk_tokens=500,
        chunk_overlap=50,
    )


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client wired to a full app: tmp data_dir, fake embedder/ollama."""
    app = create_app(make_test_settings(tmp_path), embedder=FakeEmbedder(), ollama=FakeOllama())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

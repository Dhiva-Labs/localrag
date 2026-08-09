"""Shared test fixtures: fast deterministic fakes standing in for real models."""

import hashlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from localrag.config import Settings
from localrag.core.generate import OllamaUnavailable
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

    def __init__(self, tokens: list[str] | None = None, *, fail_after: int | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["This ", "is ", "a ", "fake ", "answer."]
        self.fail_after = fail_after
        self.call_count = 0

    async def is_up(self) -> bool:
        return True

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        self.call_count += 1
        for i, token in enumerate(self.tokens):
            if self.fail_after is not None and i >= self.fail_after:
                raise OllamaUnavailable("simulated failure mid-stream")
            yield token


def parse_sse(text: str) -> list[tuple[str, object]]:
    """Parse "event: x\\ndata: y\\n\\n" blocks into [(event, decoded_json), ...]."""
    events: list[tuple[str, object]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name = None
        data_raw = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_raw = line[len("data:") :].strip()
        if event_name is not None and data_raw is not None:
            events.append((event_name, json.loads(data_raw)))
    return events


def make_test_settings(data_dir: Path, *, score_threshold: float = 0.30) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8090,
        data_dir=data_dir,
        ollama_url="http://localhost:11434",
        model="test-model",
        embed_model="test-embed-model",
        top_k=5,
        score_threshold=score_threshold,
        chunk_tokens=500,
        chunk_overlap=50,
    )


@pytest.fixture
def client_factory(tmp_path: Path):
    """Build a test client with overridable fakes.

    Use this directly (instead of the plain `client` fixture) when a test
    needs a handle on the fake ollama/embedder instance itself -- e.g. to
    assert a call count or configure a mid-stream failure.
    """

    @asynccontextmanager
    async def _factory(
        *, embedder: Any = None, ollama: Any = None, score_threshold: float = 0.30
    ) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(
            make_test_settings(tmp_path, score_threshold=score_threshold),
            embedder=embedder if embedder is not None else FakeEmbedder(),
            ollama=ollama if ollama is not None else FakeOllama(),
        )
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac

    return _factory


@pytest.fixture
async def client(client_factory) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client wired to a full app: tmp data_dir, fake embedder/ollama."""
    async with client_factory() as ac:
        yield ac

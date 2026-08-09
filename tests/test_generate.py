"""Tests for localrag.core.generate. Must not import server/api modules. No real network."""

import httpx
import pytest

from localrag.core.generate import REFUSAL, OllamaClient, OllamaUnavailable, build_prompt
from localrag.core.store import RetrievedChunk


def test_build_prompt_contains_question_filenames_pages_texts_and_refusal() -> None:
    chunks = [
        RetrievedChunk(
            doc_id="a", filename="solar.txt", page=1, text="solar panel content", score=0.9
        ),
        RetrievedChunk(
            doc_id="b", filename="hydro.md", page=2, text="hydro dam content", score=0.8
        ),
    ]

    prompt = build_prompt("how do solar panels work", chunks)

    assert "how do solar panels work" in prompt
    assert "solar.txt" in prompt
    assert "hydro.md" in prompt
    assert "p. 1" in prompt
    assert "p. 2" in prompt
    assert "solar panel content" in prompt
    assert "hydro dam content" in prompt
    assert REFUSAL in prompt


async def test_stream_generate_yields_response_fragments_until_done() -> None:
    body = (
        '{"response": "Hello", "done": false}\n'
        '{"response": "", "done": false}\n'
        '{"response": " world", "done": false}\n'
        '{"response": "", "done": true}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, content=body.encode("utf-8"))

    client = OllamaClient(
        "http://localhost:11434", "test-model", transport=httpx.MockTransport(handler)
    )

    fragments = [token async for token in client.stream_generate("hello")]

    assert fragments == ["Hello", " world"]


async def test_stream_generate_stops_after_done_and_ignores_trailing_lines() -> None:
    body = (
        '{"response": "first", "done": false}\n'
        '{"response": "", "done": true}\n'
        '{"response": "should not appear", "done": false}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    client = OllamaClient(
        "http://localhost:11434", "test-model", transport=httpx.MockTransport(handler)
    )

    fragments = [token async for token in client.stream_generate("hello")]

    assert fragments == ["first"]


async def test_is_up_true_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    client = OllamaClient(
        "http://localhost:11434", "test-model", transport=httpx.MockTransport(handler)
    )

    assert await client.is_up() is True


async def test_is_up_false_on_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient(
        "http://localhost:11434", "test-model", transport=httpx.MockTransport(handler)
    )

    assert await client.is_up() is False


async def test_stream_generate_raises_ollama_unavailable_on_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient(
        "http://localhost:11434", "test-model", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(OllamaUnavailable):
        async for _ in client.stream_generate("hello"):
            pass


async def test_stream_generate_client_has_unbounded_read_timeout() -> None:
    client = OllamaClient("http://localhost:11434", "test-model")

    async with client._stream_client() as http_client:
        assert http_client.timeout.read is None
        assert http_client.timeout.connect == 5.0

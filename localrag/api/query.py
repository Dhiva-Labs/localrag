"""Streaming question-answer endpoint: POST /api/query."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from localrag.core.generate import REFUSAL, OllamaUnavailable, build_prompt
from localrag.core.retriever import retrieve
from localrag.core.store import RetrievedChunk

if TYPE_CHECKING:
    from localrag.server import Deps

router = APIRouter(prefix="/api")

_TOP_K_MIN = 1
_TOP_K_MAX = 50


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _dedupe_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], RetrievedChunk] = {}
    for chunk in chunks:
        key = (chunk.filename, chunk.page)
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk

    ordered = sorted(best.values(), key=lambda chunk: chunk.score, reverse=True)
    return [
        {
            "doc": chunk.filename,
            "page": chunk.page,
            "snippet": chunk.text[:200],
            "score": round(chunk.score, 3),
        }
        for chunk in ordered
    ]


async def _event_stream(question: str, top_k: int, deps: Deps) -> AsyncIterator[str]:
    chunks = retrieve(
        question,
        embedder=deps.embedder,
        store=deps.store,
        top_k=top_k,
        score_threshold=deps.settings.score_threshold,
    )

    if not chunks:
        # no chunk cleared the threshold: refuse without ever calling Ollama
        yield _sse_event("token", {"text": REFUSAL})
        yield _sse_event("citations", [])
        yield _sse_event("done", {})
        return

    prompt = build_prompt(question, chunks)
    try:
        async for fragment in deps.ollama.stream_generate(prompt):
            yield _sse_event("token", {"text": fragment})
    except (OllamaUnavailable, httpx.HTTPError) as exc:
        yield _sse_event("error", {"message": str(exc)})
        yield _sse_event("done", {})
        return

    yield _sse_event("citations", _dedupe_citations(chunks))
    yield _sse_event("done", {})


@router.post("/query")
async def query_endpoint(body: QueryRequest, request: Request) -> StreamingResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank")

    deps: Deps = request.app.state.deps
    top_k = body.top_k if body.top_k is not None else deps.settings.top_k
    top_k = max(_TOP_K_MIN, min(_TOP_K_MAX, top_k))

    return StreamingResponse(
        _event_stream(body.question, top_k, deps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

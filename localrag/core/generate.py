"""Prompt assembly and a streaming Ollama client."""

import json
from collections.abc import AsyncIterator

import httpx

from localrag.core.store import RetrievedChunk

REFUSAL = "I don't know based on the indexed documents."

# a cold model load in Ollama can take much longer than the 5s httpx default
# before the first token arrives, so only the connect phase stays bounded --
# the read side waits as long as Ollama takes and ends when it sends done.
_STREAM_TIMEOUT = httpx.Timeout(5.0, read=None)

_INSTRUCTIONS = (
    "Answer the question using only the information in the context below. "
    f'If the answer is not present in the context, reply with exactly: "{REFUSAL}" '
    "Be concise. Never mention the context, the source documents, or the numbered "
    "blocks in your answer -- just answer the question directly."
)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{i}] {chunk.filename} (p. {chunk.page}):\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return f"{_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self, base_url: str, model: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport

    def _stream_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=_STREAM_TIMEOUT)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2},
        }
        try:
            async with (
                self._stream_client() as client,
                client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response,
            ):
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    fragment = data.get("response", "")
                    if fragment:
                        yield fragment
                    if data.get("done"):
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailable(f"ollama unavailable at {self.base_url}") from exc

    async def is_up(self) -> bool:
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

"""FastAPI application factory and the shared dependency container."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from localrag.config import Settings
from localrag.core.embedder import Embedder
from localrag.core.generate import OllamaClient
from localrag.core.store import Store


@dataclass
class Deps:
    settings: Settings
    store: Store
    embedder: Any
    ollama: Any


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Any = None,
    ollama: Any = None,
) -> FastAPI:
    resolved_settings = settings if settings is not None else Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = Store(resolved_settings.data_dir)
        active_embedder = embedder
        if active_embedder is None:
            active_embedder = Embedder(resolved_settings.embed_model)

        active_ollama = ollama
        if active_ollama is None:
            active_ollama = OllamaClient(resolved_settings.ollama_url, resolved_settings.model)

        app.state.deps = Deps(
            settings=resolved_settings,
            store=store,
            embedder=active_embedder,
            ollama=active_ollama,
        )
        yield

    app = FastAPI(lifespan=lifespan)

    from localrag.api.ingest import router as ingest_router
    from localrag.api.query import router as query_router

    app.include_router(ingest_router)
    app.include_router(query_router)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        deps: Deps = app.state.deps

        try:
            docs = deps.store.count()
            chroma_ok = True
        except Exception:
            docs = 0
            chroma_ok = False

        try:
            ollama_ok = await deps.ollama.is_up()
        except Exception:
            ollama_ok = False

        return {
            "status": "ok" if chroma_ok and ollama_ok else "degraded",
            "chroma": chroma_ok,
            "ollama": ollama_ok,
            "model": deps.settings.model,
            "docs": docs,
        }

    return app

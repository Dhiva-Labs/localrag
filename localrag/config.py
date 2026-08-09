"""Environment-driven runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    ollama_url: str
    model: str
    embed_model: str
    top_k: int
    score_threshold: float
    chunk_tokens: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=os.environ.get("LOCALRAG_HOST", "127.0.0.1"),
            port=_env_int("LOCALRAG_PORT", 8090),
            data_dir=Path(os.environ.get("LOCALRAG_DATA_DIR", "./data")),
            ollama_url=os.environ.get("LOCALRAG_OLLAMA_URL", "http://localhost:11434"),
            model=os.environ.get("LOCALRAG_MODEL", "llama3.2"),
            embed_model=os.environ.get("LOCALRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            top_k=_env_int("LOCALRAG_TOP_K", 5),
            score_threshold=_env_float("LOCALRAG_SCORE_THRESHOLD", 0.30),
            chunk_tokens=_env_int("LOCALRAG_CHUNK_TOKENS", 500),
            chunk_overlap=_env_int("LOCALRAG_CHUNK_OVERLAP", 50),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid integer value for {name}: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid float value for {name}: {raw!r}") from exc

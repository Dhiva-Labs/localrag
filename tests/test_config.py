"""Tests for localrag.config."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from localrag.config import Settings

_ENV_VARS = [
    "LOCALRAG_HOST",
    "LOCALRAG_PORT",
    "LOCALRAG_DATA_DIR",
    "LOCALRAG_OLLAMA_URL",
    "LOCALRAG_MODEL",
    "LOCALRAG_EMBED_MODEL",
    "LOCALRAG_TOP_K",
    "LOCALRAG_SCORE_THRESHOLD",
    "LOCALRAG_CHUNK_TOKENS",
    "LOCALRAG_CHUNK_OVERLAP",
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_when_no_env_vars_set() -> None:
    settings = Settings.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8090
    assert settings.data_dir == Path("./data")
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.model == "llama3.2"
    assert settings.embed_model == "BAAI/bge-small-en-v1.5"
    assert settings.top_k == 5
    assert settings.score_threshold == 0.30
    assert settings.chunk_tokens == 500
    assert settings.chunk_overlap == 50


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALRAG_HOST", "0.0.0.0")
    monkeypatch.setenv("LOCALRAG_PORT", "9000")
    monkeypatch.setenv("LOCALRAG_DATA_DIR", "/tmp/custom-data")
    monkeypatch.setenv("LOCALRAG_TOP_K", "10")
    monkeypatch.setenv("LOCALRAG_SCORE_THRESHOLD", "0.55")

    settings = Settings.from_env()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.data_dir == Path("/tmp/custom-data")
    assert settings.top_k == 10
    assert settings.score_threshold == 0.55


def test_invalid_int_env_raises_value_error_naming_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALRAG_PORT", "not-a-number")

    with pytest.raises(ValueError, match="LOCALRAG_PORT"):
        Settings.from_env()


def test_invalid_float_env_raises_value_error_naming_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALRAG_SCORE_THRESHOLD", "not-a-float")

    with pytest.raises(ValueError, match="LOCALRAG_SCORE_THRESHOLD"):
        Settings.from_env()


def test_settings_is_frozen() -> None:
    settings = Settings.from_env()

    with pytest.raises(FrozenInstanceError):
        settings.host = "changed"

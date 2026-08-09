"""Tests for localrag.core.embedder. Never instantiate the real model here."""

import sys

from localrag.core.embedder import QUERY_PREFIX, Embedder


def test_module_import_does_not_load_sentence_transformers() -> None:
    assert "sentence_transformers" not in sys.modules


def test_query_prefix_exact_value() -> None:
    assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


def test_construction_does_not_load_model() -> None:
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert embedder._model is None

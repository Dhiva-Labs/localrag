"""Tests for localrag.core.retriever. Must not import server/api modules."""

import hashlib
from pathlib import Path

from conftest import FakeEmbedder

from localrag.core.chunker import Chunk
from localrag.core.extract import ExtractedDoc, Page
from localrag.core.retriever import retrieve
from localrag.core.store import Store

EMBEDDER = FakeEmbedder()


def _seed(store: Store, filename: str, seed: str, chunk_texts: list[str]) -> None:
    sha256 = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    doc = ExtractedDoc(filename=filename, sha256=sha256, pages=[Page(number=1, text=seed)])
    chunks = [Chunk(text=text, page=1, index=i) for i, text in enumerate(chunk_texts)]
    embeddings = EMBEDDER.embed_texts(chunk_texts)
    store.add_document(doc, chunks, embeddings)


def test_retrieve_ranks_matching_vocabulary_chunk_first(tmp_path: Path) -> None:
    store = Store(tmp_path)
    _seed(store, "solar.txt", "solar seed", ["solar panels convert sunlight into electricity"])
    _seed(store, "hydro.txt", "hydro seed", ["dam turbines generate power from flowing water"])

    results = retrieve(
        "how do solar panels convert sunlight into electricity",
        embedder=EMBEDDER,
        store=store,
        top_k=5,
        score_threshold=0.0,
    )

    assert len(results) > 0
    assert "solar" in results[0].text


def test_retrieve_filters_below_score_threshold(tmp_path: Path) -> None:
    store = Store(tmp_path)
    _seed(store, "solar.txt", "solar seed", ["solar panels convert sunlight into electricity"])

    results = retrieve(
        "zebra giraffe elephant rhinoceros hippopotamus",
        embedder=EMBEDDER,
        store=store,
        top_k=5,
        score_threshold=0.99,
    )

    assert results == []


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    store = Store(tmp_path)
    _seed(
        store,
        "mixed.txt",
        "mixed seed",
        [
            "alpha bravo charlie delta",
            "echo foxtrot golf hotel",
            "india juliet kilo lima",
        ],
    )

    results = retrieve(
        "alpha bravo charlie delta",
        embedder=EMBEDDER,
        store=store,
        top_k=1,
        score_threshold=0.0,
    )

    assert len(results) == 1

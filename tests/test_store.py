"""Tests for localrag.core.store, using FakeEmbedder vectors and tmp_path."""

import hashlib
import json
from pathlib import Path

from conftest import FakeEmbedder

from localrag.core.chunker import Chunk
from localrag.core.extract import ExtractedDoc, Page
from localrag.core.store import Store

EMBEDDER = FakeEmbedder()


def _make_doc(
    filename: str, seed: str, chunk_texts: list[str]
) -> tuple[ExtractedDoc, list[Chunk], list[list[float]]]:
    sha256 = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    doc = ExtractedDoc(filename=filename, sha256=sha256, pages=[Page(number=1, text=seed)])
    chunks = [Chunk(text=text, page=1, index=i) for i, text in enumerate(chunk_texts)]
    embeddings = EMBEDDER.embed_texts(chunk_texts)
    return doc, chunks, embeddings


def test_add_document_then_list_docs_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc, chunks, embeddings = _make_doc(
        "solar.txt",
        "solar seed",
        ["solar panels need regular cleaning", "inverter checks matter too"],
    )

    info = store.add_document(doc, chunks, embeddings)

    assert info.doc_id == doc.sha256[:12]
    assert info.filename == "solar.txt"
    assert info.sha256 == doc.sha256
    assert info.pages == 1
    assert info.chunks == 2

    docs = store.list_docs()
    assert len(docs) == 1
    assert docs[0] == info


def test_find_by_hash_hit_and_miss(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc, chunks, embeddings = _make_doc("wind.txt", "wind seed", ["wind turbines spin steadily"])
    store.add_document(doc, chunks, embeddings)

    found = store.find_by_hash(doc.sha256)
    assert found is not None
    assert found.doc_id == doc.sha256[:12]

    assert store.find_by_hash("0" * 64) is None


def test_find_by_hash_enables_caller_side_dedupe(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc, chunks, embeddings = _make_doc("dup.txt", "dup seed", ["duplicate content example"])
    first_info = store.add_document(doc, chunks, embeddings)

    existing = store.find_by_hash(doc.sha256)
    assert existing is not None
    assert existing.doc_id == first_info.doc_id
    assert store.count() == 1


def test_delete_removes_registry_entry_and_vectors(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc, chunks, embeddings = _make_doc(
        "hydro.txt", "hydro seed", ["dam turbines generate steady power"]
    )
    info = store.add_document(doc, chunks, embeddings)

    assert store.delete(info.doc_id) is True
    assert store.list_docs() == []
    assert store.find_by_hash(doc.sha256) is None

    query_vec = EMBEDDER.embed_query("dam turbines generate steady power")
    assert store.query(query_vec, top_k=5) == []


def test_delete_unknown_id_returns_false(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.delete("nonexistent") is False


def test_query_orders_by_score_desc_with_correct_fields(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc, chunks, embeddings = _make_doc(
        "mixed.txt",
        "mixed seed",
        [
            "solar panels convert sunlight into electricity",
            "battery storage enclosures require ventilation",
        ],
    )
    store.add_document(doc, chunks, embeddings)

    query_vec = EMBEDDER.embed_query("solar panels convert sunlight into electricity")
    results = store.query(query_vec, top_k=5)

    assert len(results) == 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert -1.0 <= r.score <= 1.0
        assert r.doc_id == doc.sha256[:12]
        assert r.filename == "mixed.txt"
        assert r.page == 1

    assert "solar" in results[0].text


def test_count_reflects_number_of_documents(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.count() == 0

    doc1, chunks1, embeddings1 = _make_doc("a.txt", "seed a", ["alpha bravo charlie"])
    store.add_document(doc1, chunks1, embeddings1)
    assert store.count() == 1

    doc2, chunks2, embeddings2 = _make_doc("b.txt", "seed b", ["delta echo foxtrot"])
    store.add_document(doc2, chunks2, embeddings2)
    assert store.count() == 2


def test_registry_persists_across_reinstantiation(tmp_path: Path) -> None:
    store1 = Store(tmp_path)
    doc, chunks, embeddings = _make_doc(
        "persist.txt", "persist seed", ["persisted content stays available"]
    )
    info = store1.add_document(doc, chunks, embeddings)

    store2 = Store(tmp_path)
    docs = store2.list_docs()
    assert len(docs) == 1
    assert docs[0].doc_id == info.doc_id

    results = store2.query(EMBEDDER.embed_query("persisted content stays available"), top_k=1)
    assert len(results) == 1


def test_list_docs_orders_by_ingested_at(tmp_path: Path) -> None:
    store = Store(tmp_path)
    doc1, chunks1, embeddings1 = _make_doc("first.txt", "seed one", ["first document content"])
    doc2, chunks2, embeddings2 = _make_doc("second.txt", "seed two", ["second document content"])
    info1 = store.add_document(doc1, chunks1, embeddings1)
    info2 = store.add_document(doc2, chunks2, embeddings2)

    docs = store.list_docs()
    assert [d.doc_id for d in docs] == [info1.doc_id, info2.doc_id]
    assert docs[0].ingested_at <= docs[1].ingested_at


def test_list_docs_orders_by_doc_id_when_ingested_at_ties(tmp_path: Path) -> None:
    registry = {
        "bbb222222222": {
            "filename": "b.txt",
            "sha256": "b" * 64,
            "pages": 1,
            "chunks": 1,
            "ingested_at": "2026-01-01T00:00:00+00:00",
        },
        "aaa111111111": {
            "filename": "a.txt",
            "sha256": "a" * 64,
            "pages": 1,
            "chunks": 1,
            "ingested_at": "2026-01-01T00:00:00+00:00",
        },
    }
    (tmp_path / "docs.json").write_text(json.dumps(registry))

    store = Store(tmp_path)
    docs = store.list_docs()

    assert [d.doc_id for d in docs] == ["aaa111111111", "bbb222222222"]

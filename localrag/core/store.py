"""Chroma-backed vector store plus a JSON document registry."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings

from localrag.core.chunker import Chunk
from localrag.core.extract import ExtractedDoc

_COLLECTION_NAME = "chunks"


@dataclass
class DocInfo:
    doc_id: str
    filename: str
    sha256: str
    pages: int
    chunks: int
    ingested_at: str


@dataclass
class RetrievedChunk:
    doc_id: str
    filename: str
    page: int
    text: str
    score: float


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.data_dir / "docs.json"

        # anonymized_telemetry must stay off: this server is offline-only and
        # must never dial out, in tests or in production.
        client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def _read_registry(self) -> dict[str, dict]:
        if not self._registry_path.exists():
            return {}
        with self._registry_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_registry(self, registry: dict[str, dict]) -> None:
        tmp_path = self._registry_path.with_name(self._registry_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self._registry_path)

    def find_by_hash(self, sha256: str) -> DocInfo | None:
        registry = self._read_registry()
        for doc_id, entry in registry.items():
            if entry["sha256"] == sha256:
                return DocInfo(doc_id=doc_id, **entry)
        return None

    def add_document(
        self, doc: ExtractedDoc, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> DocInfo:
        doc_id = doc.sha256[:12]

        if chunks:
            self._collection.add(
                ids=[f"{doc_id}:{chunk.index}" for chunk in chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "page": chunk.page,
                        "index": chunk.index,
                    }
                    for chunk in chunks
                ],
                documents=[chunk.text for chunk in chunks],
            )

        info = DocInfo(
            doc_id=doc_id,
            filename=doc.filename,
            sha256=doc.sha256,
            pages=len(doc.pages),
            chunks=len(chunks),
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )

        registry = self._read_registry()
        registry[doc_id] = {
            "filename": info.filename,
            "sha256": info.sha256,
            "pages": info.pages,
            "chunks": info.chunks,
            "ingested_at": info.ingested_at,
        }
        self._write_registry(registry)
        return info

    def list_docs(self) -> list[DocInfo]:
        registry = self._read_registry()
        infos = [DocInfo(doc_id=doc_id, **entry) for doc_id, entry in registry.items()]
        infos.sort(key=lambda info: (info.ingested_at, info.doc_id))
        return infos

    def delete(self, doc_id: str) -> bool:
        registry = self._read_registry()
        if doc_id not in registry:
            return False
        del registry[doc_id]
        self._write_registry(registry)
        self._collection.delete(where={"doc_id": doc_id})
        return True

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result["ids"][0]
        if not ids:
            return []

        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]

        retrieved = []
        for metadata, text, distance in zip(metadatas, documents, distances, strict=True):
            retrieved.append(
                RetrievedChunk(
                    doc_id=metadata["doc_id"],
                    filename=metadata["filename"],
                    page=metadata["page"],
                    text=text,
                    score=1 - distance,
                )
            )
        return retrieved

    def count(self) -> int:
        return len(self._read_registry())

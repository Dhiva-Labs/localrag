"""Document ingest pipeline plus the /api/ingest and /api/docs routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from localrag.core.chunker import chunk_pages
from localrag.core.extract import EmptyDocument, ExtractedDoc, UnsupportedFileType, extract

if TYPE_CHECKING:
    from localrag.server import Deps

router = APIRouter(prefix="/api")


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    deduped: bool


def ingest_bytes(data: bytes, filename: str, deps: Deps) -> IngestResult:
    extracted = extract(data, filename)
    return _ingest_extracted(extracted, deps)


def _ingest_extracted(extracted: ExtractedDoc, deps: Deps) -> IngestResult:
    existing = deps.store.find_by_hash(extracted.sha256)
    if existing is not None:
        return IngestResult(
            doc_id=existing.doc_id,
            filename=existing.filename,
            pages=existing.pages,
            chunks=existing.chunks,
            deduped=True,
        )

    chunks = chunk_pages(
        extracted.pages,
        max_tokens=deps.settings.chunk_tokens,
        overlap=deps.settings.chunk_overlap,
        count_tokens=deps.embedder.count_tokens,
    )
    embeddings = deps.embedder.embed_texts([chunk.text for chunk in chunks]) if chunks else []
    info = deps.store.add_document(extracted, chunks, embeddings)
    return IngestResult(
        doc_id=info.doc_id,
        filename=info.filename,
        pages=info.pages,
        chunks=info.chunks,
        deduped=False,
    )


@router.post("/ingest")
async def ingest_endpoint(request: Request, files: Annotated[list[UploadFile], File()]) -> dict:
    deps: Deps = request.app.state.deps

    extracted_docs = []
    for upload in files:
        filename = upload.filename or ""
        data = await upload.read()
        try:
            extracted_docs.append(extract(data, filename))
        except (UnsupportedFileType, EmptyDocument) as exc:
            raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc

    results = [_ingest_extracted(doc, deps) for doc in extracted_docs]
    return {
        "documents": [
            {
                "doc_id": r.doc_id,
                "filename": r.filename,
                "pages": r.pages,
                "chunks": r.chunks,
                "deduped": r.deduped,
            }
            for r in results
        ]
    }


@router.get("/docs")
async def list_docs_endpoint(request: Request) -> dict:
    deps: Deps = request.app.state.deps
    docs = deps.store.list_docs()
    return {
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "sha256": d.sha256,
                "pages": d.pages,
                "chunks": d.chunks,
                "ingested_at": d.ingested_at,
            }
            for d in docs
        ]
    }


@router.delete("/docs/{doc_id}", status_code=204)
async def delete_doc_endpoint(doc_id: str, request: Request) -> None:
    deps: Deps = request.app.state.deps
    if not deps.store.delete(doc_id):
        raise HTTPException(status_code=404, detail=f"unknown document id: {doc_id}")

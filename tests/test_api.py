"""API integration tests: httpx.AsyncClient + ASGITransport against the real app."""

from pathlib import Path

import httpx

from localrag.core.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


async def test_ingest_txt_returns_200_with_doc_shape(client: httpx.AsyncClient) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()

    resp = await client.post("/api/ingest", files={"files": ("sample.txt", data, "text/plain")})

    assert resp.status_code == 200
    documents = resp.json()["documents"]
    assert len(documents) == 1
    doc = documents[0]
    assert doc["filename"] == "sample.txt"
    assert doc["pages"] == 1
    assert doc["chunks"] > 0
    assert doc["deduped"] is False
    assert "doc_id" in doc


async def test_ingest_pdf_reports_two_pages(client: httpx.AsyncClient) -> None:
    data = (FIXTURES / "sample.pdf").read_bytes()

    resp = await client.post(
        "/api/ingest", files={"files": ("sample.pdf", data, "application/pdf")}
    )

    assert resp.status_code == 200
    doc = resp.json()["documents"][0]
    assert doc["pages"] == 2


async def test_ingest_multiple_files_in_one_request(client: httpx.AsyncClient) -> None:
    txt = (FIXTURES / "sample.txt").read_bytes()
    md = (FIXTURES / "sample.md").read_bytes()

    resp = await client.post(
        "/api/ingest",
        files=[
            ("files", ("sample.txt", txt, "text/plain")),
            ("files", ("sample.md", md, "text/markdown")),
        ],
    )

    assert resp.status_code == 200
    documents = resp.json()["documents"]
    assert len(documents) == 2
    assert {d["filename"] for d in documents} == {"sample.txt", "sample.md"}


async def test_reupload_same_bytes_is_deduped_and_not_duplicated(
    client: httpx.AsyncClient,
) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()

    first = await client.post("/api/ingest", files={"files": ("sample.txt", data, "text/plain")})
    second = await client.post("/api/ingest", files={"files": ("sample.txt", data, "text/plain")})

    assert first.json()["documents"][0]["deduped"] is False
    assert second.json()["documents"][0]["deduped"] is True
    assert first.json()["documents"][0]["doc_id"] == second.json()["documents"][0]["doc_id"]

    docs = (await client.get("/api/docs")).json()["documents"]
    assert len(docs) == 1


async def test_ingest_unsupported_extension_returns_400_naming_file(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post("/api/ingest", files={"files": ("notes.csv", b"a,b,c", "text/csv")})

    assert resp.status_code == 400
    assert "notes.csv" in resp.json()["detail"]


async def test_ingest_rejects_whole_request_when_any_file_is_bad(
    client: httpx.AsyncClient,
) -> None:
    txt = (FIXTURES / "sample.txt").read_bytes()

    resp = await client.post(
        "/api/ingest",
        files=[
            ("files", ("sample.txt", txt, "text/plain")),
            ("files", ("bad.csv", b"x,y", "text/csv")),
        ],
    )

    assert resp.status_code == 400
    assert "bad.csv" in resp.json()["detail"]
    docs = (await client.get("/api/docs")).json()["documents"]
    assert docs == []


async def test_get_docs_shape(client: httpx.AsyncClient) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    await client.post("/api/ingest", files={"files": ("sample.txt", data, "text/plain")})

    resp = await client.get("/api/docs")

    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert len(docs) == 1
    for key in ("doc_id", "filename", "sha256", "pages", "chunks", "ingested_at"):
        assert key in docs[0]


async def test_delete_then_get_shows_gone_unknown_id_404(client: httpx.AsyncClient) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    ingest_resp = await client.post(
        "/api/ingest", files={"files": ("sample.txt", data, "text/plain")}
    )
    doc_id = ingest_resp.json()["documents"][0]["doc_id"]

    delete_resp = await client.delete(f"/api/docs/{doc_id}")
    assert delete_resp.status_code == 204

    docs = (await client.get("/api/docs")).json()["documents"]
    assert docs == []

    missing_resp = await client.delete(f"/api/docs/{doc_id}")
    assert missing_resp.status_code == 404


async def test_health_with_fakes(client: httpx.AsyncClient) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    await client.post("/api/ingest", files={"files": ("sample.txt", data, "text/plain")})

    resp = await client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chroma"] is True
    assert body["ollama"] is True
    assert body["model"] == "test-model"
    assert body["docs"] == 1


async def test_ingest_then_delete_fresh_store_sees_zero_docs(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    ingest_resp = await client.post(
        "/api/ingest", files={"files": ("sample.txt", data, "text/plain")}
    )
    doc_id = ingest_resp.json()["documents"][0]["doc_id"]
    await client.delete(f"/api/docs/{doc_id}")

    fresh_store = Store(tmp_path)
    assert fresh_store.count() == 0

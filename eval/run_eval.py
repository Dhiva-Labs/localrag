"""Measure retrieval and refusal behaviour on the golden set.

Runs the real ingest, chunking, embedding, vector store and `retrieve()` code, the
same path /api/query uses before it decides to refuse. Ollama is not involved:
LocalRAG's embeddings come from sentence-transformers, and a refusal is decided by
the score threshold before any generation. Generated answer text is NOT scored.

    pip install -e ".[dev]"
    python eval/run_eval.py [--k 5] [--threshold 0.30] [--json out.json]

The numbers depend on the embedder (LOCALRAG_EMBED_MODEL), so the report always
states which one produced them. The first run downloads the embedding model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import date
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localrag.api.ingest import ingest_bytes  # noqa: E402
from localrag.config import Settings  # noqa: E402
from localrag.core.embedder import Embedder  # noqa: E402
from localrag.core.retriever import retrieve  # noqa: E402
from localrag.core.store import Store  # noqa: E402
from localrag.server import Deps  # noqa: E402

HERE = Path(__file__).parent


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _summary(scores: list[float]) -> str:
    if not scores:
        return "n/a"
    return f"min {min(scores):.3f}  median {statistics.median(scores):.3f}  max {max(scores):.3f}"


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=settings.top_k, help="top-k (default: setting)")
    parser.add_argument(
        "--threshold", type=float, default=settings.score_threshold, help="refusal threshold"
    )
    parser.add_argument("--golden", type=Path, default=HERE / "golden.json")
    parser.add_argument("--docs", type=Path, default=HERE / "docs")
    parser.add_argument("--json", type=Path, help="also write the raw per-question results here")
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    embedder = Embedder(settings.embed_model)

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp))
        deps = Deps(settings=settings, store=store, embedder=embedder, ollama=None)
        for path in sorted(args.docs.iterdir()):
            if path.is_file():
                res = ingest_bytes(path.read_bytes(), path.name, deps)
                print(f"ingested {res.filename}: {res.pages} pages, {res.chunks} chunks")

        rows = []
        for item in golden:
            q = item["question"]
            # unthresholded top-k: did the right page come back at all?
            raw = store.query(embedder.embed_query(q), args.k)
            # the real path, as /api/query calls it: refused iff nothing clears the threshold
            kept = retrieve(
                q, embedder=embedder, store=store, top_k=args.k, score_threshold=args.threshold
            )
            top = raw[0].score if raw else 0.0
            row = {
                "id": item["id"],
                "question": q,
                "answerable": item["answerable"],
                "top_score": round(top, 4),
                "refused": not kept,
                "top": [(c.filename, c.page, round(c.score, 3)) for c in raw],
            }
            if item["answerable"]:
                want = (item["doc"], item["page"])
                row["expected"] = want
                row["hit_any_score"] = any((c.filename, c.page) == want for c in raw)
                row["hit_after_threshold"] = any((c.filename, c.page) == want for c in kept)
            rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    hits = sum(r["hit_any_score"] for r in ans)
    hits_thr = sum(r["hit_after_threshold"] for r in ans)
    wrongful = [r for r in ans if r["refused"]]
    correct = [r for r in unans if r["refused"]]
    missed = [r for r in unans if not r["refused"]]

    print()
    print(f"date:       {date.today().isoformat()}")
    print(f"commit:     {_commit()}")
    st_version = _ver("sentence-transformers")
    print(f"embedder:   {settings.embed_model} (sentence-transformers {st_version})")
    print(f"chromadb:   {_ver('chromadb')}")
    print(
        f"settings:   top_k={args.k} threshold={args.threshold} "
        f"chunk_tokens={settings.chunk_tokens} overlap={settings.chunk_overlap}"
    )
    print(f"questions:  {len(ans)} answerable, {len(unans)} unanswerable")
    print()
    print(f"retrieval hit rate @{args.k} (any score):   {hits}/{len(ans)} = {hits / len(ans):.1%}")
    print(
        f"retrieval hit rate @{args.k} (>= threshold): "
        f"{hits_thr}/{len(ans)} = {hits_thr / len(ans):.1%}"
    )
    print(f"wrongful refusals:  {len(wrongful)}/{len(ans)} = {len(wrongful) / len(ans):.1%}")
    print(f"correct refusals:   {len(correct)}/{len(unans)} = {len(correct) / len(unans):.1%}")
    print()
    print("top-1 score, answerable:   ", _summary([r["top_score"] for r in ans]))
    print("top-1 score, unanswerable: ", _summary([r["top_score"] for r in unans]))

    if len(ans) - hits:
        print("\nretrieval misses:")
        for r in ans:
            if not r["hit_any_score"]:
                print(f"  {r['id']} {r['question']!r} expected {r['expected']} got {r['top'][:3]}")
    if wrongful:
        print("\nwrongful refusals:")
        for r in wrongful:
            print(f"  {r['id']} {r['question']!r} top score {r['top_score']}")
    if missed:
        print("\nanswered when it should have refused:")
        for r in missed:
            print(f"  {r['id']} {r['question']!r} top score {r['top_score']}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

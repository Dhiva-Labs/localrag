"""Command-line entry point: run the server or ingest documents directly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localrag.config import Settings

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def main() -> int:
    parser = argparse.ArgumentParser(prog="localrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="run the web server")

    ingest_parser = subparsers.add_parser("ingest", help="ingest a file or directory")
    ingest_parser.add_argument("path", type=Path)

    args = parser.parse_args()

    if args.command == "serve":
        return _serve()
    return _ingest(args.path)


def _serve() -> int:
    import uvicorn

    from localrag.server import create_app

    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


def _ingest(path: Path) -> int:
    from localrag.api.ingest import ingest_bytes
    from localrag.core.embedder import Embedder
    from localrag.core.store import Store
    from localrag.server import Deps

    if not path.exists():
        print(f"path not found: {path}", file=sys.stderr)
        return 1

    files = _collect_files(path)
    if not files:
        print(f"no supported files found under {path}", file=sys.stderr)
        return 1

    settings = Settings.from_env()
    deps = Deps(
        settings=settings,
        store=Store(settings.data_dir),
        embedder=Embedder(settings.embed_model),
        ollama=None,
    )

    ingested = 0
    for file_path in files:
        try:
            data = file_path.read_bytes()
            result = ingest_bytes(data, file_path.name, deps)
        except Exception as exc:
            print(f"{file_path.name}: skipped ({exc})")
            continue

        if result.deduped:
            print(f"{file_path.name} -> {result.doc_id} (deduplicate hit)")
        else:
            print(
                f"{file_path.name} -> {result.doc_id} "
                f"({result.pages} pages, {result.chunks} chunks)"
            )
        ingested += 1

    print(f"ingested {ingested}/{len(files)} files")
    return 0 if ingested > 0 else 1


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


if __name__ == "__main__":
    sys.exit(main())

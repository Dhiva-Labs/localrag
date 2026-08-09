"""Turn raw uploaded bytes into paged, hashed text ready for chunking."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

_SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


@dataclass
class Page:
    number: int
    text: str


@dataclass
class ExtractedDoc:
    filename: str
    sha256: str
    pages: list[Page]


def extract(data: bytes, filename: str) -> ExtractedDoc:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(data)
    elif suffix in _SUPPORTED_TEXT_SUFFIXES:
        pages = _extract_text(data)
    else:
        raise UnsupportedFileType(f"unsupported file type: {suffix or filename!r}")

    if not any(page.text.strip() for page in pages):
        raise EmptyDocument(f"no extractable text in {filename}")

    digest = hashlib.sha256(data).hexdigest()
    return ExtractedDoc(filename=filename, sha256=digest, pages=pages)


def _extract_pdf(data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[Page] = []
    for number, raw_page in enumerate(reader.pages, start=1):
        text = (raw_page.extract_text() or "").strip()
        if text:
            # true page number is preserved even though blank pages are dropped
            pages.append(Page(number=number, text=text))
    return pages


def _extract_text(data: bytes) -> list[Page]:
    text = data.decode("utf-8", errors="replace")
    return [Page(number=1, text=text)]

"""Tests for localrag.core.extract. Must stay independent of the chunker."""

import hashlib
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from localrag.core.extract import EmptyDocument, UnsupportedFileType, extract

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pdf(page_texts: list[str | None]) -> bytes:
    """Build a tiny in-memory PDF; None means a genuinely content-free page."""
    writer = PdfWriter()
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_ref = writer._add_object(font)

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is not None:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            ops = f"BT /F1 14 Tf 72 700 Td ({escaped}) Tj ET"
            stream = DecodedStreamObject()
            stream.set_data(ops.encode("latin-1"))
            content_ref = writer._add_object(stream)
            resources = DictionaryObject()
            resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
            page[NameObject("/Resources")] = resources
            page[NameObject("/Contents")] = content_ref

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_extracts_two_pages_with_expected_sentences() -> None:
    data = (FIXTURES / "sample.pdf").read_bytes()
    doc = extract(data, "sample.pdf")

    assert [p.number for p in doc.pages] == [1, 2]
    assert "solar panels" in doc.pages[0].text.lower()
    assert "battery storage" in doc.pages[1].text.lower()


def test_pdf_skips_blank_pages_but_keeps_true_page_numbers() -> None:
    data = _make_pdf(["First page text here.", None, "Third page text here."])
    doc = extract(data, "doc.pdf")

    assert [p.number for p in doc.pages] == [1, 3]
    assert doc.pages[0].text == "First page text here."
    assert doc.pages[1].text == "Third page text here."


def test_txt_extracts_single_page() -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    doc = extract(data, "sample.txt")

    assert len(doc.pages) == 1
    assert doc.pages[0].number == 1
    assert "wind turbines" in doc.pages[0].text.lower()


def test_md_extracts_single_page() -> None:
    data = (FIXTURES / "sample.md").read_bytes()
    doc = extract(data, "sample.md")

    assert len(doc.pages) == 1
    assert doc.pages[0].number == 1
    assert "hydro power" in doc.pages[0].text.lower()


def test_txt_decodes_invalid_utf8_with_replacement() -> None:
    data = b"valid text \xff\xfe more text"
    doc = extract(data, "broken.txt")

    assert len(doc.pages) == 1
    assert "�" in doc.pages[0].text


def test_sha256_matches_hash_of_raw_bytes() -> None:
    data = (FIXTURES / "sample.txt").read_bytes()
    doc = extract(data, "sample.txt")

    assert doc.sha256 == hashlib.sha256(data).hexdigest()


def test_sha256_is_stable_across_calls() -> None:
    data = (FIXTURES / "sample.md").read_bytes()
    first = extract(data, "sample.md")
    second = extract(data, "sample.md")

    assert first.sha256 == second.sha256


@pytest.mark.parametrize("filename", ["notes.csv", "archive.zip", "README"])
def test_unsupported_extension_raises(filename: str) -> None:
    with pytest.raises(UnsupportedFileType):
        extract(b"some content", filename)


def test_empty_file_raises_empty_document() -> None:
    with pytest.raises(EmptyDocument):
        extract(b"", "sample.txt")


def test_whitespace_only_file_raises_empty_document() -> None:
    with pytest.raises(EmptyDocument):
        extract(b"   \n\n   \t  ", "sample.md")


def test_all_blank_pdf_raises_empty_document() -> None:
    data = _make_pdf([None, None])
    with pytest.raises(EmptyDocument):
        extract(data, "blank.pdf")

"""Tests for localrag.core.chunker, using a trivial word-count token counter."""

from localrag.core.chunker import chunk_pages
from localrag.core.extract import Page


def count_tokens(text: str) -> int:
    return len(text.split())


def _shared_boundary_words(prev_text: str, curr_text: str) -> int:
    """Largest k such that the last k words of prev equal the first k of curr."""
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    for k in range(min(len(prev_words), len(curr_words)), 0, -1):
        if prev_words[-k:] == curr_words[:k]:
            return k
    return 0


def test_short_page_produces_single_chunk() -> None:
    page = Page(number=1, text="A short paragraph that easily fits in one chunk.")

    chunks = chunk_pages([page], max_tokens=50, overlap=5, count_tokens=count_tokens)

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].index == 0


def test_long_page_packs_within_budget_with_sequential_indexes() -> None:
    paragraphs = [f"Paragraph {i} covers a distinct topic for chunking tests." for i in range(20)]
    text = "\n\n".join(paragraphs)
    page = Page(number=7, text=text)

    chunks = chunk_pages([page], max_tokens=30, overlap=5, count_tokens=count_tokens)

    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 30
        assert chunk.page == 7
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_never_cross_page_boundary() -> None:
    page1 = Page(number=1, text="Alpha bravo charlie delta echo foxtrot golf hotel.")
    page2 = Page(number=2, text="India juliet kilo lima mike november oscar papa.")

    chunks = chunk_pages([page1, page2], max_tokens=50, overlap=5, count_tokens=count_tokens)

    assert {c.page for c in chunks} == {1, 2}
    for chunk in chunks:
        if chunk.page == 1:
            assert "india" not in chunk.text.lower()
        else:
            assert "alpha" not in chunk.text.lower()


def test_oversize_paragraph_splits_by_sentence_with_overlap() -> None:
    sentences = [f"Sentence {i} states a simple fact." for i in range(10)]
    paragraph = " ".join(sentences)
    page = Page(number=3, text=paragraph)

    chunks = chunk_pages([page], max_tokens=20, overlap=6, count_tokens=count_tokens)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page == 3
        assert count_tokens(chunk.text) <= 20
    for prev_chunk, curr_chunk in zip(chunks, chunks[1:], strict=False):
        assert _shared_boundary_words(prev_chunk.text, curr_chunk.text) > 0


def test_overlap_tokens_roughly_respected() -> None:
    sentences = [f"Sentence {i} states a simple fact." for i in range(10)]
    paragraph = " ".join(sentences)
    page = Page(number=1, text=paragraph)
    overlap = 6

    chunks = chunk_pages([page], max_tokens=20, overlap=overlap, count_tokens=count_tokens)

    shared = _shared_boundary_words(chunks[0].text, chunks[1].text)
    assert 0 < shared <= overlap


def test_oversize_single_sentence_uses_hard_window_split_with_overlap() -> None:
    words = [f"word{i}" for i in range(60)]
    sentence = " ".join(words)  # no sentence punctuation: one giant run-on sentence
    page = Page(number=5, text=sentence)

    chunks = chunk_pages([page], max_tokens=15, overlap=4, count_tokens=count_tokens)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page == 5
        assert count_tokens(chunk.text) <= 15
    for prev_chunk, curr_chunk in zip(chunks, chunks[1:], strict=False):
        assert _shared_boundary_words(prev_chunk.text, curr_chunk.text) > 0


def test_empty_pages_produce_no_chunks() -> None:
    pages = [
        Page(number=1, text=""),
        Page(number=2, text="   \n\n  "),
        Page(number=3, text="Real content lives on this page only."),
    ]

    chunks = chunk_pages(pages, max_tokens=50, overlap=5, count_tokens=count_tokens)

    assert [c.page for c in chunks] == [3]


def test_zero_overlap_produces_no_shared_boundary_words() -> None:
    paragraphs = [f"Paragraph {i} has some unique filler content words." for i in range(10)]
    text = "\n\n".join(paragraphs)
    page = Page(number=1, text=text)

    chunks = chunk_pages([page], max_tokens=20, overlap=0, count_tokens=count_tokens)

    assert len(chunks) > 1
    for prev_chunk, curr_chunk in zip(chunks, chunks[1:], strict=False):
        assert _shared_boundary_words(prev_chunk.text, curr_chunk.text) == 0

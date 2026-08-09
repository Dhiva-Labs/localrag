"""Token-aware chunking that keeps chunks inside a single page."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from localrag.core.extract import Page

_PARAGRAPH_RE = re.compile(r"\n\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    page: int
    index: int


def chunk_pages(
    pages: list[Page],
    *,
    max_tokens: int = 500,
    overlap: int = 50,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        for chunk_text in _chunk_text(text, max_tokens, overlap, count_tokens):
            chunks.append(Chunk(text=chunk_text, page=page.number, index=index))
            index += 1
    return chunks


def _chunk_text(
    text: str, max_tokens: int, overlap: int, count_tokens: Callable[[str], int]
) -> list[str]:
    pieces: list[str] = []
    for paragraph in _split_paragraphs(text):
        pieces.extend(_fit_paragraph(paragraph, max_tokens, overlap, count_tokens))

    # Pack to a reduced budget so the best-effort overlap prepended afterward
    # still fits under max_tokens for the common many-small-pieces case.
    pack_budget = max(1, max_tokens - overlap)
    packed = _pack_greedy(pieces, pack_budget, count_tokens)
    return _apply_overlap(packed, max_tokens, overlap, count_tokens)


def _fit_paragraph(
    paragraph: str, max_tokens: int, overlap: int, count_tokens: Callable[[str], int]
) -> list[str]:
    if count_tokens(paragraph) <= max_tokens:
        return [paragraph]

    pieces: list[str] = []
    for sentence in _split_sentences(paragraph):
        if count_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_window_split(sentence, max_tokens, overlap, count_tokens))
    return pieces


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text.strip()) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _hard_window_split(
    text: str, max_tokens: int, overlap: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """Slide a fixed token window over an oversize sentence, overlap baked in.

    Packing pieces together can't help here since one sentence is already
    over budget alone, so the window itself must carry the overlap.
    """
    words = text.split()
    windows: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = start + 1
        while end < n and count_tokens(" ".join(words[start : end + 1])) <= max_tokens:
            end += 1
        window_text = " ".join(words[start:end])
        windows.append(window_text)
        if end >= n:
            break
        trailing = _trailing_snippet(window_text, overlap, count_tokens)
        step_back = len(trailing.split())
        start = max(start + 1, end - step_back)
    return windows


def _pack_greedy(pieces: list[str], budget: int, count_tokens: Callable[[str], int]) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = current + [piece]
        if current and count_tokens("\n\n".join(candidate)) > budget:
            packed.append("\n\n".join(current))
            current = [piece]
        else:
            current = candidate
    if current:
        packed.append("\n\n".join(current))
    return packed


def _apply_overlap(
    packed: list[str], max_tokens: int, overlap: int, count_tokens: Callable[[str], int]
) -> list[str]:
    if overlap <= 0 or len(packed) < 2:
        return packed

    result = [packed[0]]
    for prev_text, curr_text in zip(packed, packed[1:], strict=False):
        trailing = _trailing_snippet(prev_text, overlap, count_tokens)
        merged = f"{trailing} {curr_text}" if trailing else curr_text
        # best-effort: skip the overlap rather than breach max_tokens
        if trailing and count_tokens(merged) <= max_tokens:
            result.append(merged)
        else:
            result.append(curr_text)
    return result


def _trailing_snippet(text: str, overlap: int, count_tokens: Callable[[str], int]) -> str:
    if overlap <= 0:
        return ""
    words = text.split()
    if not words:
        return ""

    tail: list[str] = []
    for word in reversed(words):
        candidate = [word, *tail]
        if tail and count_tokens(" ".join(candidate)) > overlap:
            break
        tail = candidate
    return " ".join(tail)

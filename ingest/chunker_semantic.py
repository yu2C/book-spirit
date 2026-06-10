"""Semantic chunking with the same BGE model used for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from core.config import (
    EMBEDDING_MODEL,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from ingest.chunker import (
    CHAPTER_PART_RE,
    CHAPTER_RE,
    Chunk,
    normalize_md_text,
    parse_line_structure,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?…])")


def _is_structure_line(
    stripped: str, chapter_update: str | None, heading_update: str | None
) -> bool:
    if chapter_update is not None and heading_update == "":
        return True
    if stripped.startswith("#"):
        return True
    return bool(CHAPTER_PART_RE.match(stripped) or CHAPTER_RE.match(stripped))


@dataclass
class _Span:
    text: str
    chapter: str = ""
    heading: str = ""


def _get_embedding_model():
    from ingest.indexer import load_embedding_model

    return load_embedding_model(EMBEDDING_MODEL)


def _split_line_sentences(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    parts = _SENTENCE_SPLIT_RE.split(stripped)
    return [p.strip() for p in parts if p.strip()]


def _iter_sentence_spans(md_text: str) -> List[_Span]:
    """Walk MD like native chunker; emit sentence-level spans with chapter metadata."""
    md_text = normalize_md_text(md_text)
    spans: List[_Span] = []
    current_chapter = ""
    current_heading = ""
    body_started = False

    for line in md_text.split("\n"):
        stripped = line.strip()
        if (
            stripped == "背景"
            or stripped.startswith("#")
            or CHAPTER_PART_RE.match(stripped)
            or CHAPTER_RE.match(stripped)
        ):
            body_started = True

        chapter_update, heading_update = parse_line_structure(
            line, allow_short_headings=body_started
        )
        if not body_started:
            chapter_update, heading_update = None, None
        if chapter_update is not None:
            current_chapter = chapter_update
            if heading_update == "":
                current_heading = ""
        if heading_update is not None and heading_update != "":
            current_heading = heading_update

        if not stripped:
            continue

        is_structure = _is_structure_line(stripped, chapter_update, heading_update)
        if is_structure:
            spans.append(_Span(stripped, current_chapter, current_heading))
            continue

        for sent in _split_line_sentences(line):
            spans.append(_Span(sent, current_chapter, current_heading))

    return spans


def _combined_length(sentences: Sequence[str]) -> int:
    return sum(len(s) for s in sentences)


def _overlap_tail(sentences: Sequence[str], overlap: int) -> List[str]:
    if overlap <= 0 or not sentences:
        return []
    tail: List[str] = []
    total = 0
    for sent in reversed(sentences):
        if total + len(sent) > overlap and tail:
            break
        tail.insert(0, sent)
        total += len(sent)
    return tail


def _flush_chunk(
    sentences: Sequence[str],
    chapter: str,
    heading: str,
    chunk_id: int,
    chunk_size: int,
) -> Chunk:
    text = "".join(sentences)
    page = chunk_id // max(1, 2000 // chunk_size) + 1 if chunk_size else 0
    return Chunk(
        text=text.strip(),
        page=page,
        heading=heading,
        chapter=chapter,
        chunk_id=chunk_id,
    )


def _dynamic_break_threshold(similarities: Sequence[float], percentile: float) -> float:
    if not similarities:
        return SEMANTIC_SIMILARITY_THRESHOLD
    pct = float(np.percentile(similarities, percentile))
    # 只切「明顯主題斷點」：取全書低相似度百分位與固定 threshold 的較低者
    return min(pct, SEMANTIC_SIMILARITY_THRESHOLD)


def chunk_markdown_semantic(
    md_text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    min_chunk_size: int = SEMANTIC_MIN_CHUNK_SIZE,
    breakpoint_percentile: float = SEMANTIC_BREAKPOINT_PERCENTILE,
    model=None,
) -> List[Chunk]:
    """
    語意分塊：相鄰句 embedding 相似度低於動態門檻時切開，並受 chunk_size 上限約束。
    使用與檢索相同的 BGE（不加 query: 前綴）。
    """
    spans = _iter_sentence_spans(md_text)
    if not spans:
        return []

    embed_model = model or _get_embedding_model()
    texts = [s.text for s in spans]
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,
    )
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    adj_sims = [float(np.dot(embeddings[i - 1], embeddings[i])) for i in range(1, len(spans))]
    break_threshold = _dynamic_break_threshold(adj_sims, breakpoint_percentile)
    effective_threshold = min(threshold, break_threshold)

    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_chapter = spans[0].chapter
    buf_heading = spans[0].heading

    for i, span in enumerate(spans):
        if not buf:
            buf = [span.text]
            buf_chapter = span.chapter
            buf_heading = span.heading
            continue

        sim = adj_sims[i - 1]
        buf_len = _combined_length(buf)
        next_len = buf_len + len(span.text)
        semantic_break = buf_len >= min_chunk_size and sim < effective_threshold
        should_split = next_len > chunk_size or semantic_break

        if should_split:
            chunks.append(_flush_chunk(buf, buf_chapter, buf_heading, len(chunks), chunk_size))
            buf = _overlap_tail(buf, overlap)
            buf.append(span.text)
            buf_chapter = span.chapter
            buf_heading = span.heading
        else:
            buf.append(span.text)
            if span.chapter:
                buf_chapter = span.chapter
            if span.heading:
                buf_heading = span.heading

    if buf:
        chunks.append(_flush_chunk(buf, buf_chapter, buf_heading, len(chunks), chunk_size))

    return chunks

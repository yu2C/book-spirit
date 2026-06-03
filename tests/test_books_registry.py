"""Tests for multi-book registry and ask scope."""

from pathlib import Path

from core.books import BOOK_SCOPE_ALL, resolve_ask_context
from ingest.books_registry import slug_from_pdf


def test_slug_from_pdf_stable():
    pdf = Path("纳瓦尔宝典-test book.pdf")
    s1 = slug_from_pdf(pdf)
    s2 = slug_from_pdf(pdf)
    assert s1 == s2
    assert "-" in s1 or s1.isalnum()


def test_resolve_ask_all_scope():
    ctx = resolve_ask_context(BOOK_SCOPE_ALL)
    assert ctx.rag_filters is None
    assert ctx.rag_enabled is True


def test_resolve_ask_single_book_filter():
    ctx = resolve_ask_context("my-book")
    assert ctx.rag_filters is not None
    assert ctx.rag_filters.book_id == "my-book"

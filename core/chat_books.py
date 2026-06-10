"""Backward-compatible re-exports — prefer core.library_scope."""

from core.library_scope import (
    BookRow,
    format_scope_label,
    indexed_books_menu,
    resolve_book_arg,
)

__all__ = ["BookRow", "format_scope_label", "indexed_books_menu", "resolve_book_arg"]

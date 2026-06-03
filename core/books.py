"""Resolve chat/API book scope: current book, all indexed books, or archived (memory-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import SearchFilters

BOOK_SCOPE_ALL = "all"


@dataclass
class AskBookContext:
    """How RAG + memory should scope a question."""

    rag_filters: Optional[SearchFilters]
    memory_book_id: Optional[str]
    rag_enabled: bool
    hint: str = ""


def resolve_ask_context(book_id: Optional[str]) -> AskBookContext:
    from ingest.books_registry import STATUS_ARCHIVED, STATUS_INDEXED, get_book

    if not book_id or book_id == BOOK_SCOPE_ALL:
        return AskBookContext(
            rag_filters=None,
            memory_book_id=None if book_id == BOOK_SCOPE_ALL else book_id,
            rag_enabled=True,
            hint="跨書檢索（所有已索引書籍）" if book_id == BOOK_SCOPE_ALL else "",
        )

    entry = get_book(book_id)
    if entry and entry.status == STATUS_ARCHIVED:
        return AskBookContext(
            rag_filters=None,
            memory_book_id=book_id,
            rag_enabled=False,
            hint=f"《{entry.book_title}》已封存：僅使用筆記，不檢索原文。",
        )

    if entry and entry.status == STATUS_INDEXED:
        return AskBookContext(
            rag_filters=SearchFilters(book_id=book_id),
            memory_book_id=book_id,
            rag_enabled=True,
            hint="",
        )

    # Unknown or pending slug: still filter by book_id if vectors exist
    return AskBookContext(
        rag_filters=SearchFilters(book_id=book_id),
        memory_book_id=book_id,
        rag_enabled=True,
        hint="",
    )

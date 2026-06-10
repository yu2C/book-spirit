"""Book scope for chat/API: RAG filters, memory-only archived books, indexed menu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.config import SearchFilters
from core.index_catalog import list_indexed_book_ids

BOOK_SCOPE_ALL = "all"

BookRow = Tuple[int, str, str]  # number, book_id, title


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

    return AskBookContext(
        rag_filters=SearchFilters(book_id=book_id),
        memory_book_id=book_id,
        rag_enabled=True,
        hint="",
    )


def indexed_books_menu() -> List[BookRow]:
    """Sorted indexed books with 1-based numbers."""
    from ingest.books_registry import get_book, load_registry

    reg = load_registry()
    rows: List[BookRow] = []
    for n, bid in enumerate(sorted(list_indexed_book_ids()), start=1):
        entry = reg.get(bid) or get_book(bid)
        title = (entry.book_title if entry else bid)[:60]
        rows.append((n, bid, title))
    return rows


def book_id_for_number(num: int) -> str | None:
    for n, bid, _ in indexed_books_menu():
        if n == num:
            return bid
    return None


def format_scope_label(book_id: str) -> str:
    if book_id == "all":
        return "所有已索引書籍"
    for n, bid, title in indexed_books_menu():
        if bid == book_id:
            return f"[{n}] {title} ({bid})"
    return book_id


def resolve_book_arg(arg: str) -> tuple[str | None, str]:
    """
    Resolve /book argument to book_id.
    Returns (book_id, user_message). book_id None means invalid.
    """
    token = arg.strip()
    if not token:
        return None, "❌ 請指定書籍編號、book_id 或 all"
    if token.lower() == "all":
        return "all", "✅ 檢索範圍 = 所有已索引書籍（/book all）"
    if token.isdigit():
        bid = book_id_for_number(int(token))
        if bid is None:
            return None, f"❌ 無此編號（請 /books 查看 1–{len(indexed_books_menu())}）"
        return bid, f"✅ 檢索範圍 = {format_scope_label(bid)}"
    from ingest.books_registry import STATUS_INDEXED, load_registry

    reg = load_registry()
    if token in reg and reg[token].status == STATUS_INDEXED:
        return token, f"✅ 檢索範圍 = {format_scope_label(token)}"
    if token in reg:
        return None, f"❌ {token} 尚未索引 → uv run python scripts/build_index.py --book {token}"
    return None, f"❌ 未知書籍 {token}（/books 列書目）"

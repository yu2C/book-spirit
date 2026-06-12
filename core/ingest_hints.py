"""Ingest-quality hints (MarkItDown vs MinerU) for eval and ask."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.config import (
    EXTRACT_BACKEND_MARKITDOWN,
    EXTRACT_BACKEND_MINERU,
    GOLDEN_PASS_WARN_THRESHOLD,
)


def _is_pdf_source(book_info: Dict[str, Any]) -> bool:
    path = str(book_info.get("source_path") or book_info.get("source_pdf") or "")
    return path.lower().endswith(".pdf")


def mineru_rebuild_command(book_id: str) -> str:
    return (
        f"EXTRACT_BACKEND=mineru uv run python scripts/build_index.py "
        f"--book {book_id} --force"
    )


def get_book_index_info(
    book_id: str | None, meta: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if not book_id or book_id == "all":
        return None
    if meta is None:
        from core.index_catalog import load_library_meta_optional

        meta = load_library_meta_optional()
    if not meta:
        return None
    return (meta.get("books") or {}).get(book_id)


def ingest_hint_for_book(
    book_id: str | None,
    *,
    meta: Optional[Dict[str, Any]] = None,
    low_retrieval: bool = False,
) -> Optional[str]:
    """Suggest MinerU rebuild when PDF was indexed with MarkItDown and retrieval is weak."""
    info = get_book_index_info(book_id, meta)
    if not info:
        return None
    backend = (info.get("extract_backend") or EXTRACT_BACKEND_MARKITDOWN).lower()
    if backend == EXTRACT_BACKEND_MINERU:
        return None
    if not _is_pdf_source(info):
        return None
    if not low_retrieval:
        return None
    return (
        f"⚠️ 書籍「{book_id}」目前為 MarkItDown 索引，檢索信心偏低。"
        f"可試 MinerU 重建：{mineru_rebuild_command(book_id)}"
    )


def golden_ingest_warnings(
    meta: Dict[str, Any],
    *,
    pass_rate: float,
    case_book_ids: Optional[List[str]] = None,
    threshold: float = GOLDEN_PASS_WARN_THRESHOLD,
) -> List[str]:
    """Warnings after --golden when pass rate is low and PDF books use MarkItDown."""
    if pass_rate >= threshold:
        return []

    books = meta.get("books") or {}
    targets: set[str] = set(case_book_ids or [])
    if not targets:
        targets = set(books.keys())

    warnings: List[str] = []
    for bid in sorted(targets):
        info = books.get(bid)
        if not info:
            continue
        backend = (info.get("extract_backend") or EXTRACT_BACKEND_MARKITDOWN).lower()
        if backend != EXTRACT_BACKEND_MARKITDOWN:
            continue
        if not _is_pdf_source(info):
            continue
        warnings.append(
            f"⚠️ [{bid}] golden {pass_rate * 100:.0f}% < {threshold * 100:.0f}% ，"
            f"且為 MarkItDown PDF 索引。建議對照 MinerU：\n"
            f"   {mineru_rebuild_command(bid)}\n"
            f"   再跑：uv run python scripts/eval.py --golden"
        )
    return warnings

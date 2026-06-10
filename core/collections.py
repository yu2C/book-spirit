"""Per-book Qdrant collection names and search scope resolution."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from core.config import COLLECTION_NAME

LEGACY_COLLECTION = COLLECTION_NAME  # 舊版單一集合 "books"


def collection_name_for_book(book_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", book_id).strip("_")[:96]
    return f"book_{safe or 'unknown'}"


def vector_search_targets(book_id: Optional[str]) -> List[Tuple[str, str]]:
    """
    Collections to query: (collection_name, scope_book_id).

    scope_book_id 用於合併結果時標註；專屬 collection 內仍保留 payload book_id。
    """
    from core.index_catalog import list_indexed_book_ids

    if book_id and book_id != "all":
        name = collection_name_for_book(book_id)
        return [(name, book_id)]

    targets: List[Tuple[str, str]] = []
    for bid in list_indexed_book_ids():
        targets.append((collection_name_for_book(bid), bid))

    # 舊版單集合 fallback（尚未 migrate 時）
    if not targets:
        return [(LEGACY_COLLECTION, "")]
    return targets


def collection_exists_for_book(client, book_id: str) -> bool:
    dedicated = collection_name_for_book(book_id)
    if client.collection_exists(dedicated):
        return True
    if client.collection_exists(LEGACY_COLLECTION):
        return True
    return False


def point_count_for_book(client, book_id: str) -> int:
    dedicated = collection_name_for_book(book_id)
    if client.collection_exists(dedicated):
        info = client.get_collection(dedicated)
        return int(getattr(info, "points_count", 0) or 0)
    if client.collection_exists(LEGACY_COLLECTION):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            result = client.count(
                collection_name=LEGACY_COLLECTION,
                count_filter=Filter(
                    must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
                ),
                exact=True,
            )
            return int(getattr(result, "count", 0) or 0)
        except Exception:
            return 0
    return 0

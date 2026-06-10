"""Unified retrieval entry points (single pass vs fallback ladder)."""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

from core.retrieve_fallback import RetrievalFallbackResult, retrieve_with_fallback

if TYPE_CHECKING:
    from core.pipeline import NativeRAG


def resolve_native_rag(backend_or_rag: Any) -> "NativeRAG":
    if hasattr(backend_or_rag, "retrieve") and hasattr(backend_or_rag, "build_prompt"):
        return backend_or_rag
    if hasattr(backend_or_rag, "native"):
        return backend_or_rag.native
    raise TypeError(f"Unsupported retriever: {type(backend_or_rag)!r}")


def search_once(
    backend_or_rag: Any,
    query: str,
    *,
    top_k: int,
    filters=None,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    planned_queries: Optional[List[str]] = None,
    scope_book_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rag = resolve_native_rag(backend_or_rag)
    return rag.retrieve(
        query,
        top_k=top_k,
        filters=filters,
        mode=mode,
        use_rerank=use_rerank,
        planned_queries=planned_queries,
        scope_book_id=scope_book_id,
    )


def search_with_fallback(
    backend_or_rag: Any,
    query: str,
    *,
    top_k: int,
    filters=None,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    planned_queries: Optional[List[str]] = None,
    scope_book_id: Optional[str] = None,
) -> RetrievalFallbackResult:
    rag = resolve_native_rag(backend_or_rag)
    return retrieve_with_fallback(
        rag,
        query,
        top_k=top_k,
        filters=filters,
        mode=mode,
        use_rerank=use_rerank,
        planned_queries=planned_queries,
        scope_book_id=scope_book_id,
    )

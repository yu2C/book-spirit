"""Retrieval fallback ladder: hybrid_rerank → widen top_k → hybrid → vector on low scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.config import (
    RETRIEVAL_MAX_ATTEMPTS,
    RETRIEVAL_MAX_TOP_K,
    RETRIEVAL_MIN_TOP1_SCORE,
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_VECTOR,
    RETRIEVAL_RETRY_ENABLE_HYBRID,
    RETRIEVAL_RETRY_TOP_K_MULTIPLIER,
    RETRIEVAL_STRATEGY_HYBRID_RERANK,
    resolve_retrieval_settings,
)
from core.retrieval import retrieval_top_k

if TYPE_CHECKING:
    from core.pipeline import NativeRAG


@dataclass
class RetrievalAttempt:
    attempt: int
    top_k: int
    mode: str
    use_rerank: bool
    strategy: str
    top1_score: float
    num_docs: int
    sufficient: bool


@dataclass
class RetrievalFallbackResult:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    attempts: List[RetrievalAttempt] = field(default_factory=list)
    final_strategy: str = ""
    low_confidence: bool = False

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "attempts": [asdict(a) for a in self.attempts],
            "final_strategy": self.final_strategy,
            "low_confidence": self.low_confidence,
        }


def doc_top_score(doc: Dict[str, Any]) -> float:
    return float(doc.get("score") or doc.get("retrieval_score") or 0.0)


def is_retrieval_sufficient(
    docs: List[Dict[str, Any]], *, min_score: float = RETRIEVAL_MIN_TOP1_SCORE
) -> bool:
    if not docs:
        return False
    return doc_top_score(docs[0]) >= min_score


def _append_strategy(
    strategies: List[tuple[int, str, bool, str]],
    k: int,
    mode: str,
    use_rerank: bool,
) -> None:
    _, _, label = resolve_retrieval_settings(retrieval_mode=mode, use_rerank=use_rerank)
    key = (k, mode, use_rerank)
    if key not in {(s[0], s[1], s[2]) for s in strategies}:
        strategies.append((k, mode, use_rerank, label))


def _build_strategies(
    effective_k: int,
    mode: str,
    use_rerank: bool,
) -> List[tuple[int, str, bool, str]]:
    """(top_k, mode, use_rerank, label) per attempt — quality-first, downgrade on low score."""
    strategies: List[tuple[int, str, bool, str]] = []
    k_wide = min(effective_k * RETRIEVAL_RETRY_TOP_K_MULTIPLIER, RETRIEVAL_MAX_TOP_K)

    if mode == RETRIEVAL_MODE_HYBRID and use_rerank:
        _append_strategy(strategies, effective_k, RETRIEVAL_MODE_HYBRID, True)
        if k_wide != effective_k:
            _append_strategy(strategies, k_wide, RETRIEVAL_MODE_HYBRID, True)
        if RETRIEVAL_RETRY_ENABLE_HYBRID:
            _append_strategy(strategies, k_wide, RETRIEVAL_MODE_HYBRID, False)
            _append_strategy(strategies, k_wide, RETRIEVAL_MODE_VECTOR, False)
    elif mode == RETRIEVAL_MODE_HYBRID:
        _append_strategy(strategies, effective_k, RETRIEVAL_MODE_HYBRID, False)
        if k_wide != effective_k:
            _append_strategy(strategies, k_wide, RETRIEVAL_MODE_HYBRID, False)
        _append_strategy(strategies, k_wide, RETRIEVAL_MODE_VECTOR, False)
    else:
        _append_strategy(strategies, effective_k, RETRIEVAL_MODE_VECTOR, False)
        if k_wide != effective_k:
            _append_strategy(strategies, k_wide, RETRIEVAL_MODE_VECTOR, False)

    return strategies[:RETRIEVAL_MAX_ATTEMPTS]


def retrieve_with_fallback(
    rag: "NativeRAG",
    query: str,
    *,
    top_k: int,
    filters=None,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    planned_queries: Optional[List[str]] = None,
    scope_book_id: Optional[str] = None,
) -> RetrievalFallbackResult:
    from core.config import RETRIEVAL_MODE, USE_RERANK

    base_mode = mode or RETRIEVAL_MODE
    base_rerank = USE_RERANK if use_rerank is None else use_rerank
    effective_k = retrieval_top_k(query, top_k)

    strategies = _build_strategies(effective_k, base_mode, base_rerank)
    attempts: List[RetrievalAttempt] = []
    docs: List[Dict[str, Any]] = []
    final_label = ""

    for idx, (tk, attempt_mode, attempt_rerank, label) in enumerate(strategies, start=1):
        docs = rag.retrieve(
            query,
            top_k=tk,
            filters=filters,
            mode=attempt_mode,
            use_rerank=attempt_rerank,
            planned_queries=planned_queries,
            scope_book_id=scope_book_id,
        )
        top1 = doc_top_score(docs[0]) if docs else 0.0
        sufficient = is_retrieval_sufficient(docs)
        attempts.append(
            RetrievalAttempt(
                attempt=idx,
                top_k=tk,
                mode=attempt_mode,
                use_rerank=attempt_rerank,
                strategy=label,
                top1_score=top1,
                num_docs=len(docs),
                sufficient=sufficient,
            )
        )
        final_label = label
        if sufficient:
            break

    low = not is_retrieval_sufficient(docs)
    return RetrievalFallbackResult(
        documents=docs,
        attempts=attempts,
        final_strategy=final_label or RETRIEVAL_STRATEGY_HYBRID_RERANK,
        low_confidence=low and bool(docs),
    )

"""Tests for core.config."""

from core.config import (
    RETRIEVAL_STRATEGY_HYBRID_RERANK,
    RETRIEVAL_STRATEGY_VECTOR,
    SearchFilters,
    resolve_retrieval_settings,
)


def test_search_filters_empty():
    assert SearchFilters().is_empty()


def test_resolve_retrieval_strategy_vector():
    mode, rerank, label = resolve_retrieval_settings(
        retrieval_strategy=RETRIEVAL_STRATEGY_VECTOR
    )
    assert mode == "vector"
    assert rerank is False
    assert label == RETRIEVAL_STRATEGY_VECTOR


def test_resolve_retrieval_strategy_hybrid_rerank():
    mode, rerank, label = resolve_retrieval_settings(
        retrieval_strategy=RETRIEVAL_STRATEGY_HYBRID_RERANK
    )
    assert mode == "hybrid"
    assert rerank is True
    assert label == RETRIEVAL_STRATEGY_HYBRID_RERANK

from core.retrieve_fallback import (
    _build_strategies,
    doc_top_score,
    is_retrieval_sufficient,
)


def test_is_retrieval_sufficient():
    assert not is_retrieval_sufficient([], min_score=0.5)
    assert not is_retrieval_sufficient([{"score": 0.4}], min_score=0.5)
    assert is_retrieval_sufficient([{"score": 0.6}], min_score=0.5)


def test_doc_top_score_prefers_score():
    assert doc_top_score({"score": 0.7, "retrieval_score": 0.2}) == 0.7


def test_build_strategies_hybrid_rerank_quality_first():
    strategies = _build_strategies(5, "hybrid", True)
    assert strategies[0] == (5, "hybrid", True, "hybrid_rerank")
    assert strategies[1][:3] == (10, "hybrid", True)
    assert strategies[2][:3] == (10, "hybrid", False)
    assert strategies[3][:3] == (10, "vector", False)


def test_build_strategies_vector_widen_only():
    strategies = _build_strategies(5, "vector", False)
    assert strategies[0][:3] == (5, "vector", False)
    assert strategies[1][:3] == (10, "vector", False)
    assert len(strategies) == 2

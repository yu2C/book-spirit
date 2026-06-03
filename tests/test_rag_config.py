"""Retrieval strategy resolution tests."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_rag_config():
    spec = importlib.util.spec_from_file_location("rag_config", ROOT / "rag_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_strategy_vector():
    cfg = load_rag_config()
    mode, rerank, label = cfg.resolve_retrieval_settings(retrieval_strategy="vector")
    assert mode == "vector"
    assert rerank is False
    assert label == "vector"


def test_resolve_strategy_hybrid_rerank():
    cfg = load_rag_config()
    mode, rerank, label = cfg.resolve_retrieval_settings(
        retrieval_strategy="hybrid_rerank"
    )
    assert mode == "hybrid"
    assert rerank is True
    assert label == "hybrid_rerank"


def test_resolve_legacy_mode_and_rerank():
    cfg = load_rag_config()
    mode, rerank, label = cfg.resolve_retrieval_settings(
        retrieval_mode="hybrid",
        use_rerank=True,
    )
    assert mode == "hybrid"
    assert rerank is True
    assert label == "hybrid_rerank"


def test_resolve_invalid_strategy():
    cfg = load_rag_config()
    with pytest.raises(ValueError):
        cfg.resolve_retrieval_settings(retrieval_strategy="invalid")

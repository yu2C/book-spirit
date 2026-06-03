"""Hybrid retrieval unit tests (RRF merge, no GPU)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_rag_hybrid():
    spec = importlib.util.spec_from_file_location("rag_hybrid", ROOT / "rag_hybrid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rrf_merge_prefers_docs_in_both_lists():
    hybrid = load_rag_hybrid()
    vector_docs = [
        {"chunk_id": 1, "text": "a", "score": 0.9},
        {"chunk_id": 2, "text": "b", "score": 0.8},
        {"chunk_id": 3, "text": "c", "score": 0.7},
    ]
    bm25_docs = [
        {"chunk_id": 2, "text": "b", "score": 12.0},
        {"chunk_id": 4, "text": "d", "score": 10.0},
        {"chunk_id": 1, "text": "a", "score": 8.0},
    ]

    merged = hybrid.rrf_merge([vector_docs, bm25_docs], top_k=3)

    assert [doc["chunk_id"] for doc in merged] == [2, 1, 4]
    assert merged[0]["score"] > merged[1]["score"] > merged[2]["score"]


def test_rrf_merge_respects_top_k():
    hybrid = load_rag_hybrid()
    list_a = [{"chunk_id": index, "text": str(index)} for index in range(5)]
    list_b = [{"chunk_id": index + 10, "text": str(index)} for index in range(5)]

    merged = hybrid.rrf_merge([list_a, list_b], top_k=2)

    assert len(merged) == 2


def test_tokenize_zh_strips_whitespace():
    hybrid = load_rag_hybrid()
    tokens = hybrid.tokenize_zh("什麼 是\n專長")
    assert tokens == ["什", "麼", "是", "專", "長"]

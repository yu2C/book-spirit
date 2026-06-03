"""Reranker unit tests (mock CrossEncoder, no GPU)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_rag_rerank():
    spec = importlib.util.spec_from_file_location("rag_rerank", ROOT / "rag_rerank.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeReranker:
    def rerank(self, query, docs, top_k):
        ranked = []
        for index, doc in enumerate(docs):
            item = doc.copy()
            item["retrieval_score"] = item.get("score", 0.0)
            item["score"] = float(item["retrieval_score"]) + (10 - index)
            item["score_source"] = "rerank"
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


def test_rerank_documents_reorders_by_rerank_score():
    rerank = load_rag_rerank()
    docs = [
        {"chunk_id": 1, "text": "a", "score": 0.9},
        {"chunk_id": 2, "text": "b", "score": 0.8},
        {"chunk_id": 3, "text": "c", "score": 0.7},
    ]

    result = rerank.rerank_documents(
        "什麼是專長？",
        docs,
        top_k=2,
        reranker=FakeReranker(),
    )

    assert len(result) == 2
    assert result[0]["chunk_id"] == 1
    assert result[0]["score_source"] == "rerank"
    assert result[0]["retrieval_score"] == 0.9
    assert result[0]["score"] > result[0]["retrieval_score"]


def test_rerank_documents_empty_input():
    rerank = load_rag_rerank()
    assert rerank.rerank_documents("test", [], top_k=3, reranker=FakeReranker()) == []

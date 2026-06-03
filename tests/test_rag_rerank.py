"""Tests for core.rerank (mocked reranker)."""

from core.rerank import rerank_documents


class _FakeReranker:
    def rerank(self, query, docs, top_k):
        out = []
        for doc in docs:
            item = doc.copy()
            item["retrieval_score"] = item.get("score", 0.0)
            item["score"] = 0.9 if item["chunk_id"] == 2 else 0.2
            item["score_source"] = "rerank"
            out.append(item)
        out.sort(key=lambda d: d["score"], reverse=True)
        return out[:top_k]


def test_rerank_documents_orders_by_score():
    docs = [
        {"text": "first", "score": 0.5, "chunk_id": 1},
        {"text": "second", "score": 0.6, "chunk_id": 2},
    ]
    out = rerank_documents("query", docs, top_k=2, reranker=_FakeReranker())
    assert out[0]["chunk_id"] == 2
    assert out[0]["score_source"] == "rerank"

"""Cross-encoder reranking for retrieved document candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from core.config import RERANKER_MODEL

_reranker: "Reranker | None" = None


class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]: ...


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []

        working_docs = []
        for doc in docs:
            item = doc.copy()
            item["retrieval_score"] = item.get("score", 0.0)
            item["score_source"] = "retrieval"
            working_docs.append(item)

        pairs = [[query, doc["text"]] for doc in working_docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(working_docs, scores):
            doc["score"] = float(score)
            doc["score_source"] = "rerank"

        ranked = sorted(working_docs, key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


def get_reranker(model_name: str = RERANKER_MODEL) -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=model_name)
    return _reranker


def rerank_documents(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int,
    reranker: RerankerProtocol | None = None,
) -> List[Dict[str, Any]]:
    if not docs:
        return []
    engine = reranker or get_reranker()
    return engine.rerank(query, docs, top_k=top_k)

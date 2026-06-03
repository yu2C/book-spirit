"""Hybrid retrieval: BM25 + vector search merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.config import (
    BM25_CORPUS_FILE,
    RETRIEVE_CANDIDATES,
    RRF_K,
    SearchFilters,
    apply_payload_filters,
)

logger = logging.getLogger(__name__)

_bm25_index: "BM25Index | None" = None


def tokenize_zh(text: str) -> List[str]:
    """Character-level tokens for Chinese BM25 (no jieba dependency)."""
    return [char for char in text.lower() if not char.isspace()]


def rrf_merge(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: int,
    id_key: str = "chunk_id",
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Merge multiple ranked doc lists with Reciprocal Rank Fusion."""
    scores: Dict[Any, float] = {}
    doc_by_id: Dict[Any, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc.get(id_key)
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_by_id:
                doc_by_id[doc_id] = doc.copy()

    merged: List[Dict[str, Any]] = []
    for doc_id in sorted(scores.keys(), key=lambda key: scores[key], reverse=True):
        item = doc_by_id[doc_id].copy()
        item["score"] = scores[doc_id]
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


class BM25Index:
    def __init__(self, records: List[Dict[str, Any]], tokenized: List[List[str]]):
        from rank_bm25 import BM25Okapi

        self.records = records
        self.tokenized = tokenized
        self.bm25 = BM25Okapi(tokenized)

    @classmethod
    def load(cls, corpus_path: Path | str = BM25_CORPUS_FILE) -> "BM25Index":
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 語料不存在: {path}，請執行 uv run python scripts/build_index.py 重建索引"
            )
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        tokenized = [tokenize_zh(record.get("text", "")) for record in records]
        return cls(records, tokenized)

    def search(
        self,
        query: str,
        top_k: int = RETRIEVE_CANDIDATES,
        filters: SearchFilters | None = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = tokenize_zh(query)
        if not query_tokens:
            return []

        candidate_indices = list(range(len(self.records)))
        if filters and not filters.is_empty():
            filtered_records = apply_payload_filters(self.records, filters)
            allowed_ids = {record.get("chunk_id") for record in filtered_records}
            candidate_indices = [
                index
                for index, record in enumerate(self.records)
                if record.get("chunk_id") in allowed_ids
            ]

        if not candidate_indices:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for index in ranked[:top_k]:
            if scores[index] <= 0:
                continue
            record = self.records[index].copy()
            record["score"] = float(scores[index])
            results.append(record)
        return results


def get_bm25_index(corpus_path: Path | str = BM25_CORPUS_FILE) -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index.load(corpus_path)
    return _bm25_index


def bm25_search(
    query: str,
    top_k: int = RETRIEVE_CANDIDATES,
    filters: SearchFilters | None = None,
    corpus_path: Path | str = BM25_CORPUS_FILE,
) -> List[Dict[str, Any]]:
    index = get_bm25_index(corpus_path)
    return index.search(query, top_k=top_k, filters=filters)


def hybrid_retrieve(
    native_rag,
    query: str,
    candidate_limit: int,
    filters: SearchFilters | None = None,
    candidates: int = RETRIEVE_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Vector + BM25 candidate retrieval merged with RRF."""
    try:
        bm25_docs = bm25_search(query, top_k=candidates, filters=filters)
    except FileNotFoundError as exc:
        logger.warning("%s — fallback to vector-only", exc)
        return native_rag.vector_search(
            query,
            limit=candidate_limit,
            filters=filters,
        )[:candidate_limit]

    vector_docs = native_rag.vector_search(query, limit=candidates, filters=filters)
    merged = rrf_merge([vector_docs, bm25_docs], top_k=candidate_limit)
    return apply_payload_filters(merged, filters)[:candidate_limit]

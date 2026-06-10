"""Hybrid retrieval: BM25 + vector search merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.config import (
    BM25_CORPUS_FILE,
    RETRIEVE_CANDIDATES,
    RRF_K,
    SearchFilters,
    apply_payload_filters,
)

logger = logging.getLogger(__name__)

_BM25_CACHE_MAX = 2
_bm25_cache: Dict[Tuple[str, ...], "BM25Index"] = {}
_bm25_cache_order: List[Tuple[str, ...]] = []


def tokenize_zh(text: str) -> List[str]:
    return [char for char in text.lower() if not char.isspace()]


def rrf_merge(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: int,
    id_key: str = "point_id",
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    scores: Dict[Any, float] = {}
    doc_by_id: Dict[Any, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc.get(id_key)
            if doc_id is None:
                doc_id = f"{doc.get('book_id')}:{doc.get('chunk_id')}"
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
    def load_paths(cls, paths: List[Path]) -> "BM25Index":
        records: List[Dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as handle:
                records.extend(json.load(handle))
        if not records:
            raise FileNotFoundError(
                "找不到 BM25 語料，請執行: uv run python scripts/build_index.py --all"
            )
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
            allowed_ids = {
                record.get("point_id") or f"{record.get('book_id')}:{record.get('chunk_id')}"
                for record in filtered_records
            }
            candidate_indices = [
                index
                for index, record in enumerate(self.records)
                if (record.get("point_id") or f"{record.get('book_id')}:{record.get('chunk_id')}")
                in allowed_ids
            ]

        if not candidate_indices:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)

        results: List[Dict[str, Any]] = []
        for index in ranked[:top_k]:
            if scores[index] <= 0:
                continue
            record = self.records[index].copy()
            record["score"] = float(scores[index])
            results.append(record)
        return results


def resolve_bm25_paths(filters: SearchFilters | None) -> List[Path]:
    from core.index_catalog import bm25_path, list_indexed_book_ids

    if filters and filters.book_id:
        return [bm25_path(filters.book_id)]
    paths = [bm25_path(bid) for bid in list_indexed_book_ids()]
    if BM25_CORPUS_FILE.exists():
        paths.append(BM25_CORPUS_FILE)
    return [p for p in paths if p.exists()]


def get_bm25_index(filters: SearchFilters | None = None) -> BM25Index:
    paths = resolve_bm25_paths(filters)
    key = tuple(str(p) for p in paths)
    if key in _bm25_cache:
        if key in _bm25_cache_order:
            _bm25_cache_order.remove(key)
        _bm25_cache_order.append(key)
        return _bm25_cache[key]
    index = BM25Index.load_paths(paths)
    _bm25_cache[key] = index
    _bm25_cache_order.append(key)
    while len(_bm25_cache_order) > _BM25_CACHE_MAX:
        old = _bm25_cache_order.pop(0)
        _bm25_cache.pop(old, None)
    return index


def bm25_search(
    query: str,
    top_k: int = RETRIEVE_CANDIDATES,
    filters: SearchFilters | None = None,
) -> List[Dict[str, Any]]:
    index = get_bm25_index(filters)
    return index.search(query, top_k=top_k, filters=filters)


def hybrid_retrieve(
    native_rag,
    query: str,
    candidate_limit: int,
    filters: SearchFilters | None = None,
    candidates: int = RETRIEVE_CANDIDATES,
    scope_book_id: str | None = None,
) -> List[Dict[str, Any]]:
    try:
        bm25_docs = bm25_search(query, top_k=candidates, filters=filters)
    except FileNotFoundError as exc:
        logger.warning("%s — fallback to vector-only", exc)
        return native_rag.vector_search(
            query,
            limit=candidate_limit,
            filters=filters,
            scope_book_id=scope_book_id,
        )[:candidate_limit]

    vector_docs = native_rag.vector_search(
        query, limit=candidates, filters=filters, scope_book_id=scope_book_id
    )
    merged = rrf_merge([vector_docs, bm25_docs], top_k=candidate_limit)
    return apply_payload_filters(merged, filters)[:candidate_limit]

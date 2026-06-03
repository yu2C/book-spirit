"""Tests for core.hybrid RRF merge."""

from core.hybrid import rrf_merge


def test_rrf_merge_prefers_both_lists():
    vector = [
        {"chunk_id": 1, "text": "a", "score": 0.9},
        {"chunk_id": 2, "text": "b", "score": 0.8},
    ]
    bm25 = [
        {"chunk_id": 2, "text": "b", "score": 1.0},
        {"chunk_id": 3, "text": "c", "score": 0.7},
    ]
    merged = rrf_merge([vector, bm25], top_k=2)
    ids = [d["chunk_id"] for d in merged]
    assert 2 in ids
    assert len(merged) == 2

"""Golden retrieval test cases — load & auto-score (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

TEST_CASES_FILE = Path(__file__).resolve().parent / "test_cases.json"


def load_test_cases(path: Path | None = None) -> List[Dict[str, Any]]:
    p = path or TEST_CASES_FILE
    if not p.is_file():
        raise FileNotFoundError(f"找不到金標題庫: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if not cases:
        raise ValueError(f"金標題庫為空: {p}")
    return cases


def chunk_matches_phrases(text: str, phrases: List[str]) -> bool:
    if not phrases:
        return False
    return any(p in text for p in phrases)


def score_case(
    case: Dict[str, Any], results: List[Dict[str, Any]], *, top_k: int = 3
) -> Dict[str, Any]:
    """
    自動評分規則（檢索 only）：
    - hit@k：top-k 中至少一塊含 must_contain_any 任一片段 → pass
    - precision@k：top-k 中有幾塊命中 / k
    - chapter_ok：任一命中的塊 chapter 含 expected_chapter（若有標）
    """
    phrases = case.get("must_contain_any") or []
    expected_chapter = (case.get("expected_chapter") or "").strip()
    hits = results[:top_k]

    per_rank = []
    relevant_in_top = 0
    for rank, doc in enumerate(hits, 1):
        text = doc.get("text") or ""
        matched = chunk_matches_phrases(text, phrases)
        if matched:
            relevant_in_top += 1
        per_rank.append(
            {
                "rank": rank,
                "score": doc.get("score", 0.0),
                "chapter": doc.get("chapter") or "",
                "matched": matched,
                "preview": text[:120].replace("\n", " "),
            }
        )

    hit_at_k = relevant_in_top > 0
    precision_at_k = relevant_in_top / top_k if top_k else 0.0
    chapter_ok = None
    if expected_chapter and relevant_in_top:
        chapter_ok = any(
            expected_chapter in (doc.get("chapter") or "")
            for doc, ok in zip(hits, [r["matched"] for r in per_rank])
            if ok
        )

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "book_id": case.get("book_id"),
        "category": case.get("category"),
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "relevant_in_top": relevant_in_top,
        "chapter_ok": chapter_ok,
        "per_rank": per_rank,
        "pass": hit_at_k,
    }


def summarize_golden(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if not n:
        return {"total": 0, "pass_rate": 0.0, "avg_precision_at_k": 0.0}
    passed = sum(1 for r in results if r.get("pass"))
    return {
        "total": n,
        "passed": passed,
        "pass_rate": passed / n,
        "avg_precision_at_k": sum(r["precision_at_k"] for r in results) / n,
    }

"""Golden retrieval test cases — load & auto-score (no LLM)."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List


def reciprocal_rank(per_rank: List[dict]) -> float:
    for row in per_rank:
        if row.get("matched"):
            return 1.0 / row["rank"]
    return 0.0


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def answer_similarity(answer: str, expected: str) -> float:
    left = normalize_text(answer)
    right = normalize_text(expected)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def answer_refused(answer: str) -> bool:
    return answer.strip().startswith("❌")


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
    case: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    top_k: int = 3,
    answer: str | None = None,
) -> Dict[str, Any]:
    """
    自動評分規則（檢索 only）：
    - hit@k：top-k 中至少一塊含 must_contain_any 任一片段 → pass
    - precision@k：top-k 中有幾塊命中 / k
    - chapter_ok：任一命中的塊 chapter 含 expected_chapter（若有標）
    - answer_score：answer 與 expected_answer 的字串相似度（heuristic baseline）
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
    mrr = reciprocal_rank(per_rank)
    expect_refusal = bool(case.get("expect_refusal"))
    refusal_correct = expect_refusal and not hit_at_k
    chapter_ok = None
    if expected_chapter and relevant_in_top:
        chapter_ok = any(
            expected_chapter in (doc.get("chapter") or "")
            for doc, ok in zip(hits, [r["matched"] for r in per_rank])
            if ok
        )

    expected_answer = case.get("expected_answer") or ""
    actual_refused = answer_refused(answer) if answer is not None else False
    answer_score = answer_similarity(answer or "", expected_answer) if answer is not None else None
    answer_pass = None
    if answer is not None:
        if expect_refusal:
            answer_pass = actual_refused
        else:
            answer_pass = bool(
                hit_at_k
                and not actual_refused
                and answer_score is not None
                and answer_score >= 0.35
            )

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "book_id": case.get("book_id"),
        "category": case.get("category"),
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "relevant_in_top": relevant_in_top,
        "chapter_ok": chapter_ok,
        "per_rank": per_rank,
        "expect_refusal": expect_refusal,
        "refusal_correct": refusal_correct,
        "answer_evaluated": answer is not None,
        "answer_refused": actual_refused,
        "answer_score": answer_score,
        "answer_pass": answer_pass,
        "pass": (not hit_at_k) if expect_refusal else hit_at_k,
    }


def summarize_golden(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if not n:
        return {
            "total": 0,
            "pass_rate": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_mrr": 0.0,
            "refusal_total": 0,
            "refusal_passed": 0,
            "refusal_rate": 0.0,
            "answer_total": 0,
            "answer_passed": 0,
            "answer_pass_rate": 0.0,
            "hallucination_rate": 0.0,
        }

    passed = sum(1 for r in results if r.get("pass"))
    refusal_cases = [r for r in results if r.get("expect_refusal")]
    refusal_passed = sum(1 for r in refusal_cases if r.get("refusal_correct"))
    refusal_total = len(refusal_cases)
    answer_rows = [r for r in results if r.get("answer_evaluated")]
    answer_passed = sum(1 for r in answer_rows if r.get("answer_pass"))
    answer_total = len(answer_rows)
    answer_pass_rate = (answer_passed / answer_total) if answer_total else 0.0
    hallucination_rate = (1 - answer_pass_rate) if answer_total else 0.0

    return {
        "total": n,
        "passed": passed,
        "pass_rate": passed / n,
        "avg_precision_at_k": sum(r["precision_at_k"] for r in results) / n,
        "avg_mrr": sum(r.get("mrr", 0.0) for r in results) / n,
        "refusal_total": refusal_total,
        "refusal_passed": refusal_passed,
        "refusal_rate": (refusal_passed / refusal_total) if refusal_total else 0.0,
        "answer_total": answer_total,
        "answer_passed": answer_passed,
        "answer_pass_rate": answer_pass_rate,
        "hallucination_rate": hallucination_rate,
    }

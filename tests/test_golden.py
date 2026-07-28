"""Golden eval scoring (no Qdrant)."""

from eval.golden import (
    answer_similarity,
    chunk_matches_phrases,
    load_test_cases,
    score_case,
    summarize_golden,
)


def test_chunk_matches_phrases():
    assert chunk_matches_phrases("专长是无法通过培训获得的知识", ["无法通过培训获得"])
    assert not chunk_matches_phrases("无关段落", ["杠杆"])


def test_score_case_hit_at_k():
    case = {
        "id": "t1",
        "question": "什么是专长？",
        "book_id": "naval-almanac",
        "must_contain_any": ["无法通过培训获得"],
        "expected_chapter": "第一章",
    }
    results = [
        {"text": "其他内容", "score": 0.9, "chapter": "序"},
        {"text": "专长是无法通过培训获得的知识", "score": 0.7, "chapter": "第一章　积累财富"},
        {"text": "无关", "score": 0.6, "chapter": ""},
    ]
    row = score_case(case, results, top_k=3)
    assert row["hit_at_k"] is True
    assert row["relevant_in_top"] == 1
    assert row["precision_at_k"] == 1 / 3
    assert row["mrr"] == 0.5
    assert row["chapter_ok"] is True


def test_load_test_cases():
    cases = load_test_cases()
    assert len(cases) == 12
    assert all(c.get("expected_answer") for c in cases)


def test_summarize_golden():
    s = summarize_golden(
        [
            {"pass": True, "precision_at_k": 0.33, "mrr": 0.5},
            {
                "pass": True,
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "expect_refusal": True,
                "refusal_correct": True,
            },
        ]
    )
    assert s["total"] == 2
    assert s["passed"] == 2
    assert s["pass_rate"] == 1.0
    assert s["avg_mrr"] == 0.25
    assert s["refusal_total"] == 1
    assert s["refusal_passed"] == 1
    assert s["refusal_rate"] == 1.0
    assert s["answer_total"] == 0
    assert s["hallucination_rate"] == 0.0

def test_score_case_refusal_expected():
    case = {
        "id": "r1",
        "question": "今天台北天气怎么样？",
        "book_id": "naval-almanac",
        "must_contain_any": [],
        "expect_refusal": True,
    }
    row = score_case(case, [], top_k=3)
    assert row["expect_refusal"] is True
    assert row["refusal_correct"] is True
    assert row["pass"] is True
    assert row["mrr"] == 0.0


def test_answer_similarity_nonzero():
    assert answer_similarity("專長是無法透過培訓取得的知識", "專長是無法透過培訓取得的知識") > 0.9


def test_score_case_answer_pass():
    case = {
        "id": "a1",
        "question": "什么是专长？",
        "book_id": "naval-almanac",
        "expected_answer": "專長是無法透過培訓取得的知識",
        "must_contain_any": ["无法通过培训获得"],
    }
    results = [{"text": "专长是无法通过培训获得的知识", "score": 0.9, "chapter": "第一章"}]
    row = score_case(case, results, top_k=1, answer="專長是無法透過培訓取得的知識")
    assert row["answer_evaluated"] is True
    assert row["answer_pass"] is True
    assert row["answer_score"] >= 0.35

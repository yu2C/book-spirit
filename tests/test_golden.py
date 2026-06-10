"""Golden eval scoring (no Qdrant)."""

from eval.golden import chunk_matches_phrases, load_test_cases, score_case, summarize_golden


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
    assert row["chapter_ok"] is True


def test_load_test_cases():
    cases = load_test_cases()
    assert len(cases) == 10
    assert all(c.get("must_contain_any") and c.get("expected_answer") for c in cases)


def test_summarize_golden():
    s = summarize_golden(
        [{"pass": True, "precision_at_k": 0.33}, {"pass": False, "precision_at_k": 0.0}]
    )
    assert s["total"] == 2
    assert s["passed"] == 1
    assert s["pass_rate"] == 0.5

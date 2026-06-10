from core.ask_flow import append_low_confidence_notice


def test_append_low_confidence_notice():
    answer = "這是回答"
    out = append_low_confidence_notice(answer, {"low_confidence": True})
    assert "檢索信心偏低" in out
    assert out.endswith(answer)


def test_skip_notice_on_error_answer():
    answer = "❌ 檢索不到"
    assert append_low_confidence_notice(answer, {"low_confidence": True}) == answer

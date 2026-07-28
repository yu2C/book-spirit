from types import SimpleNamespace
from unittest.mock import patch

from core.ask_flow import append_low_confidence_notice
from core.ask_orchestrator import run_ask


def test_append_low_confidence_notice():
    answer = "這是回答"
    out = append_low_confidence_notice(answer, {"low_confidence": True})
    assert "檢索信心偏低" in out
    assert out.endswith(answer)


def test_skip_notice_on_error_answer():
    answer = "❌ 檢索不到"
    assert append_low_confidence_notice(answer, {"low_confidence": True}) == answer


class FakeRAG:
    def build_prompt(self, query, context, **kwargs):
        return "prompt"

    def generate_with_ollama(self, prompt, temperature=0.7):
        return "should not be called"

    def get_last_llm_usage(self):
        return None


def test_run_ask_rejects_when_retrieval_is_empty():
    fake_result = SimpleNamespace(
        documents=[],
        low_confidence=True,
        to_debug_dict=lambda: {"attempts": [], "low_confidence": True},
    )

    with patch("core.ask_orchestrator.search_with_fallback", return_value=fake_result), patch(
        "core.ask_orchestrator.ingest_hint_for_book", return_value=None
    ):
        result = run_ask(
            FakeRAG(),
            "整本書在講什麼？",
            backend_label="native",
        )

    assert result["answer"].startswith("❌")
    assert result["sources"] == []
    assert result["llm_time"] == 0.0
    assert result["stage_timings"]["llm_seconds"] == 0.0
    assert result["stage_timings"]["prompt_build_seconds"] == 0.0
    assert result["token_usage"] == {}

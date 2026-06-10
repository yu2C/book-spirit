"""LangGraph ask path (requires langchain group)."""

import pytest

langgraph = pytest.importorskip("langgraph")


class _FakeNative:
    def retrieve(self, *args, **kwargs):
        return [
            {
                "text": "專長是累積而來。",
                "score": 0.8,
                "chapter": "第一章",
                "heading": "專長",
                "page": 1,
                "book_id": "naval-almanac",
                "book_title": "纳瓦尔宝典",
                "chunk_id": 1,
            }
        ]

    def build_prompt(self, question, documents, **kwargs):
        return f"Q:{question}"

    def generate_with_ollama(self, prompt, temperature=0.7):
        return "這是測試回答"


def test_langgraph_ask_runs_graph(monkeypatch):
    from integrations.langgraph import LangGraphRAG

    monkeypatch.setattr("integrations.langgraph.NativeRAG", lambda **k: _FakeNative())
    rag = LangGraphRAG()
    monkeypatch.setattr(
        "core.ask_flow.prepare_ask",
        lambda *a, **k: __import__("core.ask_flow", fromlist=["PreparedAsk"]).PreparedAsk(
            question="什麼是專長",
            top_k=3,
            search_filters=None,
            effective_k=3,
            rag_enabled=True,
        ),
    )
    out = rag.ask("什麼是專長", book_id="naval-almanac")
    assert out["backend"] == "langgraph"
    assert out["answer"] == "這是測試回答"
    assert len(out["sources"]) == 1


def test_langgraph_empty_retrieval(monkeypatch):
    from integrations.langgraph import LangGraphRAG

    empty_native = _FakeNative()
    empty_native.retrieve = lambda *a, **k: []  # type: ignore[method-assign]
    monkeypatch.setattr("integrations.langgraph.NativeRAG", lambda **k: empty_native)
    rag = LangGraphRAG()
    monkeypatch.setattr(
        "core.ask_flow.prepare_ask",
        lambda *a, **k: __import__("core.ask_flow", fromlist=["PreparedAsk"]).PreparedAsk(
            question="xyz",
            top_k=3,
            search_filters=None,
            effective_k=3,
            rag_enabled=True,
        ),
    )
    out = rag.ask("xyz")
    assert out["backend"] == "langgraph"
    assert "❌" in out["answer"]
    assert out["sources"] == []

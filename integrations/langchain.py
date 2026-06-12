"""LangChain backend alias — retrieval/generation via native pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from core.ask_orchestrator import run_ask
from core.config import DEFAULT_TOP_K, SearchFilters


class LangChainRAG:
    """Thin wrapper for resume/demo; shares native retrieval and Ollama generation."""

    def __init__(self, **native_kwargs):
        from core.pipeline import NativeRAG

        self.native = NativeRAG(**native_kwargs)

    def check_ollama_health(self) -> bool:
        return self.native.check_ollama_health()

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        filters: SearchFilters | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
        *,
        planned_queries: List[str] | None = None,
        scope_book_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        return self.native.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
            planned_queries=planned_queries,
            scope_book_id=scope_book_id,
        )

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.7,
        filters: SearchFilters | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
        *,
        use_memory: bool = True,
        book_id: str | None = None,
    ) -> Dict[str, Any]:
        return run_ask(
            self.native,
            question,
            backend_label="langchain",
            top_k=top_k,
            temperature=temperature,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
            use_memory=use_memory,
            book_id=book_id,
        )

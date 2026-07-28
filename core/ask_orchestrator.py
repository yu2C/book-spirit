"""Shared ask orchestration for native, LangChain, and LangGraph backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from core.ask_flow import (
    PreparedAsk,
    append_low_confidence_notice,
    build_ask_result,
    empty_retrieval_answer,
    prepare_ask,
)
from core.config import DEFAULT_TOP_K, SearchFilters
from core.ingest_hints import ingest_hint_for_book
from core.retrieval_service import search_with_fallback


class AskRAG(Protocol):
    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        *,
        memory_block: str = "",
        user_profile: str = "",
        planner_notes: str = "",
    ) -> str: ...

    def generate_with_ollama(self, prompt: str, temperature: float = 0.7) -> str: ...

    def get_last_llm_usage(self) -> Dict[str, Any] | None: ...


@dataclass
class RetrievalOutcome:
    documents: List[Dict[str, Any]]
    retrieval_debug: Optional[Dict[str, Any]]
    ingest_hint: Optional[str]
    retrieve_time: float


def execute_retrieval(
    rag: Any,
    prepared: PreparedAsk,
    *,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    book_id: Optional[str] = None,
) -> RetrievalOutcome:
    if not prepared.rag_enabled:
        return RetrievalOutcome(
            [],
            {"attempts": [], "low_confidence": True},
            None,
            0.0,
        )

    retrieve_start = time.time()
    fb = search_with_fallback(
        rag,
        prepared.question,
        top_k=prepared.effective_k,
        filters=prepared.search_filters,
        mode=mode,
        use_rerank=use_rerank,
        planned_queries=prepared.retrieval_queries_used or None,
        scope_book_id=prepared.scope_book_id or book_id,
    )
    retrieve_time = time.time() - retrieve_start
    scope = prepared.scope_book_id or book_id
    hint = ingest_hint_for_book(
        scope,
        low_retrieval=fb.low_confidence or not fb.documents,
    )
    return RetrievalOutcome(fb.documents, fb.to_debug_dict(), hint, retrieve_time)


def execute_empty(prepared: PreparedAsk, ingest_hint: Optional[str]) -> str:
    answer = empty_retrieval_answer(prepared)
    if ingest_hint:
        answer = f"{answer}\n{ingest_hint}"
    return answer


def execute_generate(
    rag: AskRAG,
    prepared: PreparedAsk,
    documents: List[Dict[str, Any]],
    *,
    temperature: float,
    retrieval_debug: Optional[Dict[str, Any]],
) -> tuple[str, float, List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    prompt_start = time.time()
    prompt = rag.build_prompt(
        prepared.question,
        documents,
        memory_block=prepared.memory_block,
        user_profile=prepared.user_profile,
        planner_notes=prepared.planner_notes,
    )
    prompt_time = time.time() - prompt_start
    llm_start = time.time()
    answer = append_low_confidence_notice(
        rag.generate_with_ollama(prompt, temperature=temperature),
        retrieval_debug,
    )
    llm_time = time.time() - llm_start
    from core.pipeline import doc_to_source

    sources = [doc_to_source(doc) for doc in documents]
    usage_getter = getattr(rag, "get_last_llm_usage", None)
    token_usage = usage_getter() if callable(usage_getter) else {}
    return answer, llm_time, sources, {
        "prompt_build_seconds": prompt_time,
        "llm_seconds": llm_time,
    }, token_usage or {}


def run_ask(
    rag: AskRAG,
    question: str,
    *,
    backend_label: str,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 0.7,
    filters: Optional[SearchFilters] = None,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    use_memory: bool = True,
    book_id: Optional[str] = None,
) -> Dict[str, Any]:
    prepared = prepare_ask(
        question,
        top_k=top_k,
        filters=filters,
        use_memory=use_memory,
        book_id=book_id,
    )
    outcome = execute_retrieval(
        rag,
        prepared,
        mode=mode,
        use_rerank=use_rerank,
        book_id=book_id,
    )

    if not outcome.documents:
        return build_ask_result(
            prepared,
            answer=execute_empty(prepared, outcome.ingest_hint),
            sources=[],
            llm_time=0.0,
            backend=backend_label,
            stage_timings={
                "retrieve_seconds": outcome.retrieve_time,
                "prompt_build_seconds": 0.0,
                "llm_seconds": 0.0,
                "total_seconds": time.time() - prepared.start_time,
            },
            retrieval_debug=outcome.retrieval_debug,
            ingest_hint=outcome.ingest_hint,
            token_usage={},
        )

    answer, llm_time, sources, generate_timings, token_usage = execute_generate(
        rag,
        prepared,
        outcome.documents,
        temperature=temperature,
        retrieval_debug=outcome.retrieval_debug,
    )
    return build_ask_result(
        prepared,
        answer=answer,
        sources=sources,
        llm_time=llm_time,
        backend=backend_label,
        stage_timings={
            "retrieve_seconds": outcome.retrieve_time,
            **generate_timings,
            "total_seconds": time.time() - prepared.start_time,
        },
        retrieval_debug=outcome.retrieval_debug,
        ingest_hint=outcome.ingest_hint,
        token_usage=token_usage,
    )

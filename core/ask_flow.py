"""Shared ask preparation (native + LangGraph)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.books import BOOK_SCOPE_ALL, resolve_ask_context
from core.config import SearchFilters
from core.query_planner import QueryPlan, build_query_plan
from core.retrieval import is_overview_question, retrieval_top_k


@dataclass
class PreparedAsk:
    question: str
    top_k: int
    search_filters: Optional[SearchFilters]
    memory_block: str = ""
    user_profile: str = ""
    planner_notes: str = ""
    memory_notes_used: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_queries_used: List[str] = field(default_factory=list)
    effective_k: int = 5
    rag_enabled: bool = True
    ctx_hint: Optional[str] = None
    scope_book_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)


def prepare_ask(
    question: str,
    *,
    top_k: int,
    filters: Optional[SearchFilters],
    use_memory: bool,
    book_id: Optional[str],
) -> PreparedAsk:
    memory_block = ""
    user_profile = ""
    planner_notes = ""
    memory_notes_used: List[Dict[str, Any]] = []
    retrieval_queries_used: List[str] = []

    ctx = resolve_ask_context(book_id)
    search_filters = filters if filters is not None and not filters.is_empty() else ctx.rag_filters
    memory_book_id = None if book_id == BOOK_SCOPE_ALL else ctx.memory_book_id

    if use_memory:
        try:
            from memory.store import ReadingMemory, format_notes_for_prompt

            memory = ReadingMemory()
            notes = memory.search_relevant(question, book_id=memory_book_id, limit=5)
            memory_block = format_notes_for_prompt(notes)
            user_profile = memory.get_profile()
            memory_notes_used = [n.to_dict() for n in notes]
        except Exception:
            pass

    plan: Optional[QueryPlan] = None
    if ctx.rag_enabled:
        plan = build_query_plan(question, book_id)
    effective_k = plan.top_k if plan and plan.top_k else retrieval_top_k(question, top_k)
    if plan:
        retrieval_queries_used = plan.effective_queries(question)
        planner_notes = plan.notes_for_generator

    return PreparedAsk(
        question=question,
        top_k=top_k,
        search_filters=search_filters,
        memory_block=memory_block,
        user_profile=user_profile,
        planner_notes=planner_notes,
        memory_notes_used=memory_notes_used,
        retrieval_queries_used=retrieval_queries_used,
        effective_k=effective_k,
        rag_enabled=ctx.rag_enabled,
        ctx_hint=ctx.hint,
        scope_book_id=book_id,
    )


def empty_retrieval_answer(prepared: PreparedAsk) -> str:
    if prepared.ctx_hint:
        return prepared.ctx_hint
    if not prepared.rag_enabled:
        return "❌ 未找到相關內容"
    if is_overview_question(prepared.question):
        return (
            "❌ 檢索不到足以概括全書的段落（常見於剛開始讀）。\n"
            "   建議改問：具體章節、人物、事件（英文書請用 idealism、DAO 等英文關鍵字）。\n"
            "   若剛更新分塊，請：build_index.py --book <id> --force"
        )
    return (
        "❌ 檢索不到可用段落（可能落在版權/致謝頁或被過濾）。\n"
        "   請改問更具體的人名、事件，或開 USE_QUERY_PLANNER=true 後 /debug 看檢索句"
    )


def build_ask_result(
    prepared: PreparedAsk,
    *,
    answer: str,
    sources: List[Dict[str, Any]],
    llm_time: float,
    backend: str,
    retrieval_debug: Optional[Dict[str, Any]] = None,
    ingest_hint: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "question": prepared.question,
        "answer": answer,
        "sources": sources,
        "memory_notes_used": prepared.memory_notes_used,
        "retrieval_queries_used": prepared.retrieval_queries_used,
        "time_elapsed": time.time() - prepared.start_time,
        "llm_time": llm_time,
        "backend": backend,
    }
    if retrieval_debug is not None:
        result["retrieval_debug"] = retrieval_debug
    if ingest_hint:
        result["ingest_hint"] = ingest_hint
    return result

"""LangGraph RAG workflow: prepare → retrieve → generate."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from core.ask_flow import (
    PreparedAsk,
    build_ask_result,
    empty_retrieval_answer,
    prepare_ask,
)
from core.config import DEFAULT_TOP_K, OLLAMA_MODEL, OLLAMA_URL, SearchFilters
from core.ingest_hints import ingest_hint_for_book
from core.pipeline import NativeRAG, doc_to_source
from core.retrieve_fallback import retrieve_with_fallback


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    temperature: float
    filters: SearchFilters | None
    retrieval_mode: str | None
    use_rerank: bool | None
    use_memory: bool
    book_id: str | None
    prepared: PreparedAsk
    documents: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    llm_time: float
    route: Literal["generate", "empty"]
    retrieval_debug: Dict[str, Any]
    ingest_hint: str


class LangGraphRAG:
    def __init__(
        self,
        ollama_model: str = OLLAMA_MODEL,
        ollama_url: str = OLLAMA_URL,
    ):
        self.native = NativeRAG(ollama_model=ollama_model, ollama_url=ollama_url)
        self.graph = self._build_graph()

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

    def _prepare_node(self, state: RAGState) -> RAGState:
        prepared = prepare_ask(
            state["question"],
            top_k=state.get("top_k", DEFAULT_TOP_K),
            filters=state.get("filters"),
            use_memory=state.get("use_memory", True),
            book_id=state.get("book_id"),
        )
        return {"prepared": prepared}

    def _retrieve_node(self, state: RAGState) -> RAGState:
        prepared = state["prepared"]
        if not prepared.rag_enabled:
            return {
                "documents": [],
                "route": "empty",
                "retrieval_debug": {"attempts": [], "low_confidence": True},
                "ingest_hint": "",
            }

        fb = retrieve_with_fallback(
            self.native,
            prepared.question,
            top_k=prepared.effective_k,
            filters=prepared.search_filters,
            mode=state.get("retrieval_mode"),
            use_rerank=state.get("use_rerank"),
            planned_queries=prepared.retrieval_queries_used or None,
            scope_book_id=prepared.scope_book_id,
        )
        docs = fb.documents
        hint = ingest_hint_for_book(
            prepared.scope_book_id,
            low_retrieval=fb.low_confidence or not docs,
        )
        route: Literal["generate", "empty"] = "generate" if docs else "empty"
        return {
            "documents": docs,
            "route": route,
            "retrieval_debug": fb.to_debug_dict(),
            "ingest_hint": hint or "",
        }

    def _generate_node(self, state: RAGState) -> RAGState:
        prepared = state["prepared"]
        documents = state.get("documents") or []
        generate_start = time.time()
        prompt = self.native.build_prompt(
            prepared.question,
            documents,
            memory_block=prepared.memory_block,
            user_profile=prepared.user_profile,
            planner_notes=prepared.planner_notes,
        )
        answer = self.native.generate_with_ollama(
            prompt,
            temperature=state.get("temperature", 0.7),
        )
        return {
            "answer": answer,
            "sources": [doc_to_source(doc) for doc in documents],
            "llm_time": time.time() - generate_start,
        }

    def _empty_node(self, state: RAGState) -> RAGState:
        prepared = state["prepared"]
        answer = empty_retrieval_answer(prepared)
        hint = state.get("ingest_hint") or ""
        if hint:
            answer = f"{answer}\n{hint}"
        return {
            "answer": answer,
            "sources": [],
            "llm_time": 0.0,
        }

    def _route_after_retrieve(self, state: RAGState) -> str:
        return state.get("route") or "empty"

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("prepare", self._prepare_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("empty", self._empty_node)
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"generate": "generate", "empty": "empty"},
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("empty", END)
        return workflow.compile()

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
        final = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "temperature": temperature,
                "filters": filters,
                "retrieval_mode": mode,
                "use_rerank": use_rerank,
                "use_memory": use_memory,
                "book_id": book_id,
            }
        )
        prepared: PreparedAsk = final["prepared"]
        return build_ask_result(
            prepared,
            answer=final.get("answer", ""),
            sources=final.get("sources") or [],
            llm_time=float(final.get("llm_time") or 0.0),
            backend="langgraph",
            retrieval_debug=final.get("retrieval_debug"),
            ingest_hint=final.get("ingest_hint") or None,
        )

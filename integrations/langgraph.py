"""LangGraph RAG workflow: prepare → retrieve → generate."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from core.ask_flow import PreparedAsk, build_ask_result, prepare_ask
from core.ask_orchestrator import execute_empty, execute_generate, execute_retrieval
from core.config import DEFAULT_TOP_K, OLLAMA_MODEL, OLLAMA_URL, SearchFilters
from core.pipeline import NativeRAG


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
        outcome = execute_retrieval(
            self.native,
            prepared,
            mode=state.get("retrieval_mode"),
            use_rerank=state.get("use_rerank"),
            book_id=state.get("book_id"),
        )
        route: Literal["generate", "empty"] = "generate" if outcome.documents else "empty"
        return {
            "documents": outcome.documents,
            "route": route,
            "retrieval_debug": outcome.retrieval_debug or {},
            "ingest_hint": outcome.ingest_hint or "",
        }

    def _generate_node(self, state: RAGState) -> RAGState:
        prepared = state["prepared"]
        documents = state.get("documents") or []
        answer, llm_time, sources = execute_generate(
            self.native,
            prepared,
            documents,
            temperature=state.get("temperature", 0.7),
            retrieval_debug=state.get("retrieval_debug"),
        )
        return {
            "answer": answer,
            "sources": sources,
            "llm_time": llm_time,
        }

    def _empty_node(self, state: RAGState) -> RAGState:
        prepared = state["prepared"]
        return {
            "answer": execute_empty(prepared, state.get("ingest_hint") or None),
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

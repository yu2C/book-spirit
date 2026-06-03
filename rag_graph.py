"""LangGraph RAG workflow: retrieve → generate."""

from __future__ import annotations

import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from rag_config import DEFAULT_TOP_K, OLLAMA_MODEL, OLLAMA_URL, SearchFilters
from rag_native import NativeRAG, doc_to_source


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    temperature: float
    filters: SearchFilters
    retrieval_mode: str | None
    use_rerank: bool | None
    documents: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    llm_time: float


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
    ) -> List[Dict[str, Any]]:
        return self.native.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
        )

    def _retrieve_node(self, state: RAGState) -> RAGState:
        docs = self.native.retrieve(
            state["question"],
            top_k=state.get("top_k", DEFAULT_TOP_K),
            filters=state.get("filters"),
            mode=state.get("retrieval_mode"),
            use_rerank=state.get("use_rerank"),
        )
        return {"documents": docs}

    def _generate_node(self, state: RAGState) -> RAGState:
        documents = state.get("documents") or []
        if not documents:
            return {
                "answer": "❌ 未找到相關內容",
                "sources": [],
                "llm_time": 0.0,
            }

        generate_start = time.time()
        prompt = self.native.build_prompt(state["question"], documents)
        answer = self.native.generate_with_ollama(
            prompt,
            temperature=state.get("temperature", 0.7),
        )
        return {
            "answer": answer,
            "sources": [doc_to_source(doc) for doc in documents],
            "llm_time": time.time() - generate_start,
        }

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.7,
        filters: SearchFilters | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        final_state = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "temperature": temperature,
                "filters": filters,
                "retrieval_mode": mode,
                "use_rerank": use_rerank,
            }
        )
        return {
            "question": question,
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "time_elapsed": time.time() - start_time,
            "llm_time": final_state.get("llm_time", 0.0),
            "backend": "langgraph",
        }

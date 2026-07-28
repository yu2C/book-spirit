"""RAG backend factory (native / LangChain / LangGraph)."""

from __future__ import annotations

import os

from core.config import (
    BACKEND_LANGCHAIN,
    BACKEND_LANGGRAPH,
    BACKEND_NATIVE,
    SUPPORTED_BACKENDS,
)
from core.pipeline import NativeRAG


def build_rag_backend(name: str | None = None):
    backend = name or os.getenv("RAG_BACKEND", BACKEND_NATIVE)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Use: {SUPPORTED_BACKENDS}")
    if backend == BACKEND_NATIVE:
        return NativeRAG()
    if backend == BACKEND_LANGCHAIN:
        from integrations.langchain import LangChainRAG

        return LangChainRAG()
    if backend == BACKEND_LANGGRAPH:
        from integrations.langgraph import LangGraphRAG

        return LangGraphRAG()
    raise ValueError(f"Unsupported backend: {backend}")

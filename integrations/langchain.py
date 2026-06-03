"""LangChain retriever layer over existing Qdrant index + BGE embeddings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

from core.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    SearchFilters,
    create_qdrant_client,
)


class BGEZhEmbeddings(HuggingFaceEmbeddings):
    """BGE 中文：索引用原文，查詢加 query: 前綴。"""

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> BGEZhEmbeddings:
    return BGEZhEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    qdrant_path=str(QDRANT_PATH),
    qdrant_url: str | None = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    embeddings: Optional[BGEZhEmbeddings] = None,
) -> Qdrant:
    client = create_qdrant_client(qdrant_path, qdrant_url)
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings or get_embeddings(),
    )


def document_to_dict(doc: Document, score: float) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "text": doc.page_content,
        "score": score,
        "chapter": metadata.get("chapter", ""),
        "heading": metadata.get("heading", ""),
        "page": metadata.get("page", 0),
        "book_title": metadata.get("book_title", ""),
    }


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    qdrant_path=str(QDRANT_PATH),
    filters: SearchFilters | None = None,
    mode: str | None = None,
    use_rerank: bool | None = None,
) -> List[Dict[str, Any]]:
    from core.pipeline import NativeRAG

    rag = NativeRAG(qdrant_path=qdrant_path, qdrant_url=QDRANT_URL)
    return rag.retrieve(
        query,
        top_k=top_k,
        filters=filters,
        mode=mode,
        use_rerank=use_rerank,
    )


class LangChainRAG:
    """LangChain retriever + native Ollama generation."""

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
    ) -> List[Dict[str, Any]]:
        return self.native.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
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
        result = self.native.ask(
            question,
            top_k=top_k,
            temperature=temperature,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
            use_memory=use_memory,
            book_id=book_id,
        )
        result["backend"] = "langchain"
        return result

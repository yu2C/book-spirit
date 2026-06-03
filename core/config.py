"""Shared RAG configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
SAMPLE_BOOKS_DIR = ROOT_DIR / "sample_books"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "books")
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", ROOT_DIR / "qdrant_storage"))
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. http://qdrant:6333 for Docker
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DATABASE_URL = os.getenv("DATABASE_URL")

RETRIEVAL_MODE_VECTOR = "vector"
RETRIEVAL_MODE_HYBRID = "hybrid"
SUPPORTED_RETRIEVAL_MODES = (RETRIEVAL_MODE_VECTOR, RETRIEVAL_MODE_HYBRID)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", RETRIEVAL_MODE_VECTOR)

RETRIEVAL_STRATEGY_VECTOR = "vector"
RETRIEVAL_STRATEGY_HYBRID = "hybrid"
RETRIEVAL_STRATEGY_HYBRID_RERANK = "hybrid_rerank"
SUPPORTED_RETRIEVAL_STRATEGIES = (
    RETRIEVAL_STRATEGY_VECTOR,
    RETRIEVAL_STRATEGY_HYBRID,
    RETRIEVAL_STRATEGY_HYBRID_RERANK,
)
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", RETRIEVAL_STRATEGY_VECTOR)
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_CORPUS_FILE = QDRANT_PATH / "bm25_corpus.json"

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
USE_RERANK = os.getenv("USE_RERANK", "false").lower() in ("1", "true", "yes")


def resolve_retrieval_settings(
    retrieval_strategy: str | None = None,
    retrieval_mode: str | None = None,
    use_rerank: bool | None = None,
) -> tuple[str, bool, str]:
    """Return (retrieval_mode, use_rerank, strategy_label)."""
    if retrieval_strategy:
        if retrieval_strategy not in SUPPORTED_RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"Unsupported retrieval_strategy: {retrieval_strategy}. "
                f"Use: {SUPPORTED_RETRIEVAL_STRATEGIES}"
            )
        if retrieval_strategy == RETRIEVAL_STRATEGY_VECTOR:
            return RETRIEVAL_MODE_VECTOR, False, retrieval_strategy
        if retrieval_strategy == RETRIEVAL_STRATEGY_HYBRID:
            return RETRIEVAL_MODE_HYBRID, False, retrieval_strategy
        return RETRIEVAL_MODE_HYBRID, True, RETRIEVAL_STRATEGY_HYBRID_RERANK

    strategy = retrieval_strategy or RETRIEVAL_STRATEGY
    if strategy in SUPPORTED_RETRIEVAL_STRATEGIES and not (retrieval_mode or use_rerank is not None):
        return resolve_retrieval_settings(retrieval_strategy=strategy)

    mode = retrieval_mode or RETRIEVAL_MODE
    if mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode: {mode}. Use: {SUPPORTED_RETRIEVAL_MODES}")

    rerank = USE_RERANK if use_rerank is None else use_rerank
    if mode == RETRIEVAL_MODE_HYBRID and rerank:
        label = RETRIEVAL_STRATEGY_HYBRID_RERANK
    elif mode == RETRIEVAL_MODE_HYBRID:
        label = RETRIEVAL_STRATEGY_HYBRID
    else:
        label = RETRIEVAL_STRATEGY_VECTOR
    return mode, rerank, label

BACKEND_NATIVE = "native"
BACKEND_LANGCHAIN = "langchain"
BACKEND_LANGGRAPH = "langgraph"
SUPPORTED_BACKENDS = (BACKEND_NATIVE, BACKEND_LANGCHAIN, BACKEND_LANGGRAPH)


@dataclass
class SearchFilters:
    """Optional metadata filters for Qdrant payload fields."""

    chapter: Optional[str] = None
    heading: Optional[str] = None
    book_title: Optional[str] = None
    book_id: Optional[str] = None

    def is_empty(self) -> bool:
        return not any((self.chapter, self.heading, self.book_title, self.book_id))


def build_qdrant_filter(filters: SearchFilters | None):
    """Build Qdrant Filter for payload substring match (requires TEXT payload index)."""
    if filters is None or filters.is_empty():
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchText

    must = []
    if filters.chapter:
        must.append(FieldCondition(key="chapter", match=MatchText(text=filters.chapter)))
    if filters.heading:
        must.append(FieldCondition(key="heading", match=MatchText(text=filters.heading)))
    if filters.book_title:
        must.append(
            FieldCondition(key="book_title", match=MatchText(text=filters.book_title))
        )
    if filters.book_id:
        from qdrant_client.models import MatchValue

        must.append(
            FieldCondition(key="book_id", match=MatchValue(value=filters.book_id))
        )
    return Filter(must=must)


def apply_payload_filters(docs: list[dict], filters: SearchFilters | None) -> list[dict]:
    """Post-filter retrieved docs by substring match (fallback / guarantee)."""
    if filters is None or filters.is_empty():
        return docs

    filtered = []
    for doc in docs:
        if filters.chapter and filters.chapter not in (doc.get("chapter") or ""):
            continue
        if filters.heading and filters.heading not in (doc.get("heading") or ""):
            continue
        if filters.book_title and filters.book_title not in (doc.get("book_title") or ""):
            continue
        if filters.book_id and doc.get("book_id") != filters.book_id:
            continue
        filtered.append(doc)
    return filtered


def create_qdrant_client(qdrant_path: str | Path | None = None, qdrant_url: str | None = None):
    """Local path (default) or remote QDRANT_URL for Docker Compose."""
    from qdrant_client import QdrantClient

    url = qdrant_url or QDRANT_URL
    if url:
        return QdrantClient(url=url)
    path = qdrant_path or QDRANT_PATH
    return QdrantClient(path=str(path))


SYSTEM_PROMPT = """你是一個知識助手，基於提供的文本內容回答問題。

回答規則：
1. 只基於提供的文本內容回答
2. 如果文本中沒有相關信息，直接說「文本中沒有相關信息」
3. 在回答中引用具體的文本段落（書中原句可保留簡體）
4. 用清晰的邏輯組織回答
5. 一律使用繁體中文（你的解釋與總結，勿混用簡體）
6. 禁止使用 Markdown（不要用 **、#、```、- 項目符號語法）；只用純文字與換行，必要時用 1. 2. 3. 或「一、二、三」分段"""

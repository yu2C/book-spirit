"""Native RAG pipeline (SentenceTransformer + Qdrant + Ollama)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
from sentence_transformers import SentenceTransformer

from core.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_PATH,
    QDRANT_URL,
    RERANK_CANDIDATES,
    RETRIEVAL_MODE,
    RETRIEVAL_MODE_HYBRID,
    SYSTEM_PROMPT,
    USE_RERANK,
    SearchFilters,
    apply_payload_filters,
    build_qdrant_filter,
    create_qdrant_client,
)


def doc_to_source(doc: Dict[str, Any]) -> Dict[str, Any]:
    source = {
        "title": doc.get("book_title", ""),
        "chapter": doc.get("chapter", ""),
        "heading": doc.get("heading", ""),
        "page": doc.get("page", 0),
        "score": doc.get("score", 0.0),
        "text_preview": doc.get("text", "")[:100],
        "chunk_id": doc.get("chunk_id"),
    }
    if "retrieval_score" in doc:
        source["retrieval_score"] = doc["retrieval_score"]
    if doc.get("score_source"):
        source["score_source"] = doc["score_source"]
    return source


def hits_to_docs(points) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        docs.append(
            {
                "text": payload.get("text", ""),
                "score": point.score,
                "chapter": payload.get("chapter", ""),
                "heading": payload.get("heading", ""),
                "page": payload.get("page", 0),
                "book_title": payload.get("book_title", ""),
                "chunk_id": payload.get("chunk_id"),
                "point_id": point.id,
            }
        )
    return docs


class NativeRAG:
    def __init__(
        self,
        qdrant_path=str(QDRANT_PATH),
        qdrant_url: str | None = QDRANT_URL,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_model: str = OLLAMA_MODEL,
        ollama_url: str = OLLAMA_URL,
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.qdrant_client = create_qdrant_client(qdrant_path, qdrant_url)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.ollama_endpoint = f"{ollama_url}/api/generate"

    def check_ollama_health(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def vector_search(
        self,
        query: str,
        limit: int = DEFAULT_TOP_K,
        filters: Optional[SearchFilters] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0].tolist()
        response = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=build_qdrant_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return apply_payload_filters(hits_to_docs(response.points), filters)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[SearchFilters] = None,
        mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        retrieval_mode = mode or RETRIEVAL_MODE
        rerank_enabled = USE_RERANK if use_rerank is None else use_rerank
        candidate_limit = RERANK_CANDIDATES if rerank_enabled else top_k

        if retrieval_mode == RETRIEVAL_MODE_HYBRID:
            from core.hybrid import hybrid_retrieve

            docs = hybrid_retrieve(
                self,
                query,
                candidate_limit=candidate_limit,
                filters=filters,
            )
        else:
            fetch_k = candidate_limit
            if filters and not filters.is_empty():
                fetch_k = max(candidate_limit, top_k * 10, 30)
            docs = self.vector_search(query, limit=fetch_k, filters=filters)[:candidate_limit]

        if rerank_enabled and docs:
            from core.rerank import rerank_documents

            return rerank_documents(query, docs, top_k=top_k)
        return docs[:top_k]

    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        *,
        memory_block: str = "",
        user_profile: str = "",
    ) -> str:
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n[來源 {i}] {doc['book_title']} - {doc['chapter']}\n"
            context_text += f"{doc['text']}\n"

        profile_block = ""
        if user_profile.strip():
            profile_block = f"\n【讀者偏好與背景（簡述）】\n{user_profile.strip()}\n"

        memory_section = f"\n{memory_block}\n" if memory_block else ""

        return f"""{SYSTEM_PROMPT}
{profile_block}{memory_section}
提供的文本內容：
{context_text}

問題：{query}

回答："""

    def generate_with_ollama(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
        }
        try:
            response = requests.post(
                self.ollama_endpoint,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            return "❌ 生成超時，請重試"
        except requests.exceptions.ConnectionError:
            return "❌ 無法連接到 Ollama。請確保執行了：ollama serve"
        except requests.RequestException as exc:
            return f"❌ 生成失敗: {exc}"

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.7,
        filters: Optional[SearchFilters] = None,
        mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        *,
        use_memory: bool = True,
        book_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        memory_notes_used: List[Dict[str, Any]] = []

        memory_block = ""
        user_profile = ""
        if use_memory:
            try:
                from memory.store import ReadingMemory, format_notes_for_prompt

                memory = ReadingMemory()
                notes = memory.search_relevant(question, book_id=book_id, limit=5)
                memory_block = format_notes_for_prompt(notes)
                user_profile = memory.get_profile()
                memory_notes_used = [n.to_dict() for n in notes]
            except Exception:
                pass

        retrieved = self.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=mode,
            use_rerank=use_rerank,
        )
        if not retrieved:
            return {
                "question": question,
                "answer": "❌ 未找到相關內容",
                "sources": [],
                "memory_notes_used": memory_notes_used,
                "time_elapsed": time.time() - start_time,
                "llm_time": 0.0,
                "backend": "native",
            }

        generate_start = time.time()
        prompt = self.build_prompt(
            question,
            retrieved,
            memory_block=memory_block,
            user_profile=user_profile,
        )
        answer = self.generate_with_ollama(prompt, temperature=temperature)
        llm_time = time.time() - generate_start

        return {
            "question": question,
            "answer": answer,
            "sources": [doc_to_source(doc) for doc in retrieved],
            "memory_notes_used": memory_notes_used,
            "time_elapsed": time.time() - start_time,
            "llm_time": llm_time,
            "backend": "native",
        }

"""Native RAG pipeline (SentenceTransformer + Qdrant + Ollama)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
from sentence_transformers import SentenceTransformer

from core.books import BOOK_SCOPE_ALL, resolve_ask_context
from core.collections import LEGACY_COLLECTION, vector_search_targets
from core.config import (
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
from core.query_planner import QueryPlan, build_query_plan
from core.retrieval import (
    expand_retrieval_queries,
    filter_boilerplate_docs,
    merge_docs_by_best_score,
    retrieval_top_k,
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
                "book_id": payload.get("book_id", ""),
                "book_title": payload.get("book_title", ""),
                "chunk_id": payload.get("chunk_id"),
                "point_id": str(point.id),
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
        *,
        scope_book_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0].tolist()

        scope_key = scope_book_id
        if filters and filters.book_id:
            scope_key = filters.book_id
        if scope_key == BOOK_SCOPE_ALL:
            scope_key = "all"
        targets = vector_search_targets(scope_key)

        merged: List[Dict[str, Any]] = []
        per_collection = max(limit, 10) if len(targets) > 1 else limit

        for cname, _bid in targets:
            if not self.qdrant_client.collection_exists(cname):
                if cname.startswith("book_") and self.qdrant_client.collection_exists(LEGACY_COLLECTION):
                    cname = LEGACY_COLLECTION
                else:
                    continue
            qfilter = build_qdrant_filter(filters)
            response = self.qdrant_client.query_points(
                collection_name=cname,
                query=query_vector,
                query_filter=qfilter,
                limit=per_collection,
                with_payload=True,
            )
            merged.extend(hits_to_docs(response.points))

        merged.sort(key=lambda d: d.get("score", 0), reverse=True)
        return apply_payload_filters(merged, filters)[:limit]

    def _retrieve_once(
        self,
        query: str,
        *,
        candidate_limit: int,
        filters: Optional[SearchFilters],
        mode: str,
        scope_book_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if mode == RETRIEVAL_MODE_HYBRID:
            from core.hybrid import hybrid_retrieve

            return hybrid_retrieve(
                self,
                query,
                candidate_limit=candidate_limit,
                filters=filters,
                scope_book_id=scope_book_id,
            )
        fetch_k = candidate_limit
        if filters and not filters.is_empty():
            fetch_k = max(candidate_limit, 30)
        return self.vector_search(
            query, limit=fetch_k, filters=filters, scope_book_id=scope_book_id
        )[:candidate_limit]

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[SearchFilters] = None,
        mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        *,
        planned_queries: Optional[List[str]] = None,
        scope_book_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        retrieval_mode = mode or RETRIEVAL_MODE
        rerank_enabled = USE_RERANK if use_rerank is None else use_rerank
        effective_top_k = retrieval_top_k(query, top_k)
        candidate_limit = RERANK_CANDIDATES if rerank_enabled else max(effective_top_k * 3, 15)

        queries = planned_queries or expand_retrieval_queries(query)
        per_query: List[List[Dict[str, Any]]] = []
        for q in queries:
            per_query.append(
                self._retrieve_once(
                    q,
                    candidate_limit=candidate_limit,
                    filters=filters,
                    mode=retrieval_mode,
                    scope_book_id=scope_book_id,
                )
            )
        docs = merge_docs_by_best_score(per_query)
        docs = filter_boilerplate_docs(docs)

        if rerank_enabled and docs:
            from core.rerank import rerank_documents

            docs = rerank_documents(query, docs, top_k=effective_top_k)
        else:
            docs = docs[:effective_top_k]
        return docs

    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        *,
        memory_block: str = "",
        user_profile: str = "",
        planner_notes: str = "",
    ) -> str:
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n[來源 {i}] {doc['book_title']} - {doc['chapter']}\n"
            context_text += f"{doc['text']}\n"

        profile_block = ""
        if user_profile.strip():
            profile_block = f"\n【讀者偏好與背景（簡述）】\n{user_profile.strip()}\n"
        planner_block = ""
        if planner_notes.strip():
            planner_block = f"\n【檢索規劃提示】\n{planner_notes.strip()}\n"

        memory_section = f"\n{memory_block}\n" if memory_block else ""

        return f"""{SYSTEM_PROMPT}
{profile_block}{planner_block}{memory_section}
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
        retrieval_queries_used: List[str] = []

        memory_block = ""
        user_profile = ""
        planner_notes = ""

        ctx = resolve_ask_context(book_id)
        if filters is not None and not filters.is_empty():
            search_filters = filters
        else:
            search_filters = ctx.rag_filters

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

        if not ctx.rag_enabled:
            retrieved = []
        else:
            retrieved = self.retrieve(
                question,
                top_k=effective_k,
                filters=search_filters,
                mode=mode,
                use_rerank=use_rerank,
                planned_queries=retrieval_queries_used or None,
                scope_book_id=book_id,
            )

        if not retrieved:
            hint = ctx.hint or "❌ 未找到相關內容"
            if ctx.rag_enabled and not ctx.hint:
                from core.retrieval import is_overview_question

                if is_overview_question(question):
                    hint = (
                        "❌ 檢索不到足以概括全書的段落（常見於剛開始讀）。\n"
                        "   建議改問：具體章節、人物、事件（英文書請用 idealism、DAO 等英文關鍵字）。\n"
                        "   若剛更新分塊，請：build_index.py --book <id> --force"
                    )
                else:
                    hint = (
                        "❌ 檢索不到可用段落（可能落在版權/致謝頁或被過濾）。\n"
                        "   請改問更具體的人名、事件，或開 USE_QUERY_PLANNER=true 後 /debug 看檢索句"
                    )
            return {
                "question": question,
                "answer": hint,
                "sources": [],
                "memory_notes_used": memory_notes_used,
                "retrieval_queries_used": retrieval_queries_used,
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
            planner_notes=planner_notes,
        )
        answer = self.generate_with_ollama(prompt, temperature=temperature)
        llm_time = time.time() - generate_start

        return {
            "question": question,
            "answer": answer,
            "sources": [doc_to_source(doc) for doc in retrieved],
            "memory_notes_used": memory_notes_used,
            "retrieval_queries_used": retrieval_queries_used,
            "time_elapsed": time.time() - start_time,
            "llm_time": llm_time,
            "backend": "native",
        }

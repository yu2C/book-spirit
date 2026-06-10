"""Native RAG pipeline (SentenceTransformer + Qdrant + Ollama)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
from sentence_transformers import SentenceTransformer

from core.ask_flow import (
    build_ask_result,
    empty_retrieval_answer,
    prepare_ask,
)
from core.books import BOOK_SCOPE_ALL
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
from core.ingest_hints import ingest_hint_for_book
from core.retrieve_fallback import retrieve_with_fallback
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
                if cname.startswith("book_") and self.qdrant_client.collection_exists(
                    LEGACY_COLLECTION
                ):
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
        prepared = prepare_ask(
            question,
            top_k=top_k,
            filters=filters,
            use_memory=use_memory,
            book_id=book_id,
        )

        retrieval_debug = None
        ingest_hint = None
        if not prepared.rag_enabled:
            retrieved: List[Dict[str, Any]] = []
        else:
            fb = retrieve_with_fallback(
                self,
                question,
                top_k=prepared.effective_k,
                filters=prepared.search_filters,
                mode=mode,
                use_rerank=use_rerank,
                planned_queries=prepared.retrieval_queries_used or None,
                scope_book_id=prepared.scope_book_id or book_id,
            )
            retrieved = fb.documents
            retrieval_debug = fb.to_debug_dict()
            scope = prepared.scope_book_id or book_id
            ingest_hint = ingest_hint_for_book(
                scope,
                low_retrieval=fb.low_confidence or not retrieved,
            )

        if not retrieved:
            answer = empty_retrieval_answer(prepared)
            if ingest_hint:
                answer = f"{answer}\n{ingest_hint}"
            return build_ask_result(
                prepared,
                answer=answer,
                sources=[],
                llm_time=0.0,
                backend="native",
                retrieval_debug=retrieval_debug,
                ingest_hint=ingest_hint,
            )

        generate_start = time.time()
        prompt = self.build_prompt(
            question,
            retrieved,
            memory_block=prepared.memory_block,
            user_profile=prepared.user_profile,
            planner_notes=prepared.planner_notes,
        )
        answer = self.generate_with_ollama(prompt, temperature=temperature)
        llm_time = time.time() - generate_start

        return build_ask_result(
            prepared,
            answer=answer,
            sources=[doc_to_source(doc) for doc in retrieved],
            llm_time=llm_time,
            backend="native",
            retrieval_debug=retrieval_debug,
            ingest_hint=ingest_hint,
        )

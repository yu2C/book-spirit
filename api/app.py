"""
第六步：FastAPI 服務
把 RAG 系統部署為 REST API

運行方法：
  pip install -r requirements-langchain.txt
  uv run python -m api

Docker Compose：
  docker compose up -d

然後訪問：
  http://127.0.0.1:8000/docs  (Swagger 文檔)
  http://127.0.0.1:8000/search  (純檢索)
  http://127.0.0.1:8000/ask   (RAG 問答，backend: native | langchain | langgraph)
"""

from __future__ import annotations

import logging
import os
import time

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.query_log import QueryLogger
from core.config import (
    BACKEND_LANGCHAIN,
    BACKEND_LANGGRAPH,
    BACKEND_NATIVE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    SUPPORTED_BACKENDS,
    SearchFilters,
    resolve_retrieval_settings,
)
from core.pipeline import NativeRAG, doc_to_source
from memory.store import ReadingMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BACKEND = os.getenv("RAG_BACKEND", BACKEND_NATIVE)
SKIP_INIT = os.getenv("RAG_SKIP_INIT") == "1"


class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    temperature: float = 0.7
    backend: str = BACKEND_NATIVE
    chapter: str | None = None
    heading: str | None = None
    book_title: str | None = None
    retrieval_mode: str | None = None
    use_rerank: bool | None = None
    retrieval_strategy: str | None = None
    use_memory: bool = True
    book_id: str | None = None
    save_note: bool = False
    note_take: str | None = None


class NoteCreateRequest(BaseModel):
    book_id: str
    my_take: str
    book_title: str = ""
    chapter: str = ""
    heading: str = ""
    quote: str = ""
    tags: str = ""
    chunk_id: int | None = None


class NoteResponse(BaseModel):
    id: int
    book_id: str
    book_title: str
    chapter: str
    heading: str
    quote: str
    my_take: str
    tags: str
    chunk_id: int | None = None
    created_at: str
    updated_at: str


class ProfileRequest(BaseModel):
    profile_text: str


class ProfileResponse(BaseModel):
    profile_text: str
    notes_count: int


class SearchRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    backend: str = BACKEND_NATIVE
    chapter: str | None = None
    heading: str | None = None
    book_title: str | None = None
    retrieval_mode: str | None = None
    use_rerank: bool | None = None
    retrieval_strategy: str | None = None


class Source(BaseModel):
    title: str
    chapter: str
    heading: str
    page: int
    score: float
    text_preview: str
    chunk_id: int | None = None
    retrieval_score: float | None = None
    score_source: str | None = None


class SearchResponse(BaseModel):
    question: str
    sources: list[Source]
    time_elapsed: float
    backend: str
    filters: dict[str, str | None] | None = None
    retrieval_mode: str
    use_rerank: bool
    retrieval_strategy: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    time_elapsed: float
    llm_time: float
    backend: str
    memory_notes_used: list[NoteResponse] = []


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    qdrant_available: bool
    postgres_available: bool
    message: str
    default_backend: str


SearchResponse.model_rebuild()
AskResponse.model_rebuild()


def build_backend(name: str):
    if name == BACKEND_NATIVE:
        return NativeRAG()
    if name == BACKEND_LANGCHAIN:
        try:
            from integrations.langchain import LangChainRAG
        except ImportError as exc:
            raise RuntimeError(
                "LangChain backend 需要: pip install -r requirements-langchain.txt"
            ) from exc
        return LangChainRAG()
    if name == BACKEND_LANGGRAPH:
        try:
            from integrations.langgraph import LangGraphRAG
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph backend 需要: pip install -r requirements-langchain.txt"
            ) from exc
        return LangGraphRAG()
    raise ValueError(f"Unsupported backend: {name}")


def init_backends() -> dict[str, object]:
    initialized: dict[str, object] = {BACKEND_NATIVE: NativeRAG()}
    for name in (BACKEND_LANGCHAIN, BACKEND_LANGGRAPH):
        try:
            initialized[name] = build_backend(name)
        except Exception as exc:
            logger.warning("⚠️  Backend '%s' 未載入: %s", name, exc)
    return initialized


app = FastAPI(
    title="Book Spirit API",
    description="本地 RAG 書籍知識助手（native / LangChain / LangGraph）",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_logger = QueryLogger()
reading_memory = ReadingMemory()

if SKIP_INIT:
    backends: dict[str, object] = {}
    logger.info("RAG_SKIP_INIT=1，跳過模型載入（測試模式）")
else:
    logger.info("啟動 Book Spirit API...")
    try:
        backends = init_backends()
        logger.info("✅ 已載入 backends: %s", ", ".join(backends.keys()))
    except Exception as exc:
        logger.error("❌ RAG 初始化失敗: %s", exc)
        backends = {}


def get_backend(name: str):
    if name not in SUPPORTED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"不支援的 backend: {name}")
    if name not in backends:
        raise HTTPException(status_code=500, detail=f"Backend 未初始化: {name}")
    return backends[name]


def filters_from_request(request: SearchRequest | AskRequest) -> SearchFilters | None:
    filters = SearchFilters(
        chapter=request.chapter,
        heading=request.heading,
        book_title=request.book_title,
    )
    return None if filters.is_empty() else filters


def filters_to_dict(filters: SearchFilters | None) -> dict[str, str | None] | None:
    if filters is None:
        return None
    return {
        "chapter": filters.chapter,
        "heading": filters.heading,
        "book_title": filters.book_title,
    }


def resolve_retrieval_from_request(
    request: SearchRequest | AskRequest,
) -> tuple[str, bool, str]:
    try:
        return resolve_retrieval_settings(
            retrieval_strategy=request.retrieval_strategy,
            retrieval_mode=request.retrieval_mode,
            use_rerank=request.use_rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def call_retrieve(
    backend,
    question: str,
    top_k: int,
    filters: SearchFilters | None,
    retrieval_mode: str,
    use_rerank: bool,
):
    if hasattr(backend, "retrieve"):
        return backend.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=retrieval_mode,
            use_rerank=use_rerank,
        )
    if hasattr(backend, "native"):
        return backend.native.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            mode=retrieval_mode,
            use_rerank=use_rerank,
        )
    raise HTTPException(status_code=500, detail="Backend 不支援 retrieve")


def call_ask(
    backend,
    question: str,
    top_k: int,
    temperature: float,
    filters: SearchFilters | None,
    retrieval_mode: str,
    use_rerank: bool,
    *,
    use_memory: bool = True,
    book_id: str | None = None,
):
    if hasattr(backend, "ask"):
        return backend.ask(
            question,
            top_k=top_k,
            temperature=temperature,
            filters=filters,
            mode=retrieval_mode,
            use_rerank=use_rerank,
            use_memory=use_memory,
            book_id=book_id,
        )
    raise HTTPException(status_code=500, detail="Backend 不支援 ask")


def note_to_response(data: dict) -> NoteResponse:
    return NoteResponse(**data)


def check_ollama(backend) -> bool:
    return backend.check_ollama_health()


def extract_result_ids(sources: list[dict]) -> list:
    ids = []
    for source in sources:
        if source.get("chunk_id") is not None:
            ids.append(source["chunk_id"])
        elif "point_id" in source:
            ids.append(source["point_id"])
    return ids


def sources_to_models(sources: list[dict]) -> list[Source]:
    return [
        Source(
            title=s["title"],
            chapter=s["chapter"],
            heading=s["heading"],
            page=s["page"],
            score=s["score"],
            text_preview=s["text_preview"],
            chunk_id=s.get("chunk_id"),
            retrieval_score=s.get("retrieval_score"),
            score_source=s.get("score_source"),
        )
        for s in sources
    ]


def log_query_async(
    endpoint: str,
    question: str,
    backend: str,
    top_k: int,
    result_ids: list,
    latency_ms: float,
    retrieval_mode: str,
    use_rerank: bool,
    filters: dict[str, str | None] | None,
):
    query_logger.log_query(
        endpoint=endpoint,
        question=question,
        backend=backend,
        top_k=top_k,
        result_ids=result_ids,
        latency_ms=latency_ms,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
        filters=filters,
    )


@app.get("/", tags=["基本"])
async def root():
    return {
        "name": "Book Spirit API",
        "version": "1.2.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "ask": "/ask",
        },
        "backends": list(backends.keys()),
        "default_backend": DEFAULT_BACKEND,
    }


@app.get("/health", tags=["基本"])
async def health() -> HealthResponse:
    if not backends:
        postgres_ok = query_logger.check_health() if query_logger.enabled else False
        return HealthResponse(
            status="error",
            ollama_available=False,
            qdrant_available=False,
            postgres_available=postgres_ok,
            message="RAG 系統初始化失敗",
            default_backend=DEFAULT_BACKEND,
        )

    probe = get_backend(DEFAULT_BACKEND)
    ollama_ok = check_ollama(probe)
    qdrant_ok = True
    postgres_ok = query_logger.check_health() if query_logger.enabled else False

    checks = [ollama_ok, qdrant_ok]
    if query_logger.enabled:
        checks.append(postgres_ok)

    status = "ok" if all(checks) else "degraded"
    message_parts = []
    if not ollama_ok:
        message_parts.append("Ollama 未運行")
    if not qdrant_ok:
        message_parts.append("Qdrant 不可用")
    if query_logger.enabled and not postgres_ok:
        message_parts.append("Postgres 不可用")
    message = ", ".join(message_parts) if message_parts else "所有服務正常"

    return HealthResponse(
        status=status,
        ollama_available=ollama_ok,
        qdrant_available=qdrant_ok,
        postgres_available=postgres_ok,
        message=message,
        default_backend=DEFAULT_BACKEND,
    )


@app.post("/search", response_model=SearchResponse, tags=["核心"])
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    backend = get_backend(request.backend)
    filters = filters_from_request(request)
    retrieval_mode, use_rerank, retrieval_strategy = resolve_retrieval_from_request(request)
    start = time.time()

    try:
        docs = call_retrieve(
            backend,
            request.question,
            request.top_k,
            filters,
            retrieval_mode,
            use_rerank,
        )
        sources = [doc_to_source(doc) for doc in docs]
        elapsed = time.time() - start
        result_ids = extract_result_ids(sources)

        background_tasks.add_task(
            log_query_async,
            "search",
            request.question,
            request.backend,
            request.top_k,
            result_ids,
            elapsed * 1000,
            retrieval_mode,
            use_rerank,
            filters_to_dict(filters),
        )

        return SearchResponse(
            question=request.question,
            sources=sources_to_models(sources),
            time_elapsed=elapsed,
            backend=request.backend,
            filters=filters_to_dict(filters),
            retrieval_mode=retrieval_mode,
            use_rerank=use_rerank,
            retrieval_strategy=retrieval_strategy,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("搜尋失敗: %s", exc)
        raise HTTPException(status_code=500, detail=f"搜尋失敗: {exc}") from exc


@app.post("/ask", response_model=AskResponse, tags=["核心"])
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    backend = get_backend(request.backend)
    filters = filters_from_request(request)
    retrieval_mode, use_rerank, retrieval_strategy = resolve_retrieval_from_request(request)
    if not check_ollama(backend):
        raise HTTPException(
            status_code=503,
            detail="Ollama 未運行。請執行：ollama serve",
        )

    try:
        result = call_ask(
            backend,
            request.question,
            request.top_k,
            request.temperature,
            filters,
            retrieval_mode,
            use_rerank,
            use_memory=request.use_memory,
            book_id=request.book_id,
        )
        result_ids = extract_result_ids(result["sources"])

        background_tasks.add_task(
            log_query_async,
            "ask",
            request.question,
            result.get("backend", request.backend),
            request.top_k,
            result_ids,
            result["time_elapsed"] * 1000,
            retrieval_mode,
            use_rerank,
            filters_to_dict(filters),
        )

        memory_used = [
            note_to_response(n) for n in result.get("memory_notes_used", [])
        ]

        if request.save_note and result.get("answer") and not str(result["answer"]).startswith("❌"):
            bid = request.book_id or "default"
            src = result["sources"][0] if result.get("sources") else {}
            reading_memory.add_note(
                book_id=bid,
                my_take=(request.note_take or result["answer"]).strip(),
                book_title=src.get("title") or "",
                chapter=src.get("chapter") or "",
                heading=src.get("heading") or "",
                quote=(src.get("text_preview") or "").strip(),
                chunk_id=src.get("chunk_id"),
            )

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources_to_models(result["sources"]),
            memory_notes_used=memory_used,
            time_elapsed=result["time_elapsed"],
            llm_time=result["llm_time"],
            backend=result.get("backend", request.backend),
        )
    except Exception as exc:
        logger.error("處理請求失敗: %s", exc)
        raise HTTPException(status_code=500, detail=f"處理失敗: {exc}") from exc


@app.post("/notes", response_model=NoteResponse, tags=["讀書記憶"])
async def create_note(request: NoteCreateRequest):
    try:
        note = reading_memory.add_note(
            book_id=request.book_id,
            my_take=request.my_take,
            book_title=request.book_title,
            chapter=request.chapter,
            heading=request.heading,
            quote=request.quote,
            tags=request.tags,
            chunk_id=request.chunk_id,
        )
        return note_to_response(note.to_dict())
    except Exception as exc:
        logger.error("新增筆記失敗: %s", exc)
        raise HTTPException(status_code=500, detail=f"新增筆記失敗: {exc}") from exc


@app.get("/notes", response_model=list[NoteResponse], tags=["讀書記憶"])
async def list_notes(
    book_id: str | None = None,
    q: str | None = None,
    limit: int = 50,
):
    try:
        if q:
            notes = reading_memory.search_relevant(q, book_id=book_id, limit=limit)
        else:
            notes = reading_memory.list_notes(book_id=book_id, limit=limit)
        return [note_to_response(n.to_dict()) for n in notes]
    except Exception as exc:
        logger.error("列出筆記失敗: %s", exc)
        raise HTTPException(status_code=500, detail=f"列出筆記失敗: {exc}") from exc


@app.get("/profile", response_model=ProfileResponse, tags=["讀書記憶"])
async def get_profile():
    return ProfileResponse(
        profile_text=reading_memory.get_profile(),
        notes_count=reading_memory.count_notes(),
    )


@app.put("/profile", response_model=ProfileResponse, tags=["讀書記憶"])
async def update_profile(request: ProfileRequest):
    reading_memory.set_profile(request.profile_text)
    return ProfileResponse(
        profile_text=reading_memory.get_profile(),
        notes_count=reading_memory.count_notes(),
    )


@app.get("/info", tags=["基本"])
async def info():
    if not backends:
        return {"status": "error"}

    return {
        "system": "Book Spirit RAG",
        "version": "1.2.0",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OLLAMA_MODEL,
        "collection": "books",
        "backends": list(backends.keys()),
        "default_backend": DEFAULT_BACKEND,
        "query_log_enabled": query_logger.enabled,
        "features": [
            "向量搜尋 /search",
            "Metadata filter（chapter / heading / book_title）",
            "Hybrid BM25 + vector（retrieval_strategy=hybrid）",
            "Cross-encoder rerank（retrieval_strategy=hybrid_rerank）",
            "RAG 問答 /ask",
            "LangChain retriever",
            "LangGraph workflow",
            "Postgres query log",
            "SQLite 讀書筆記 /notes、/profile",
            "來源引用",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

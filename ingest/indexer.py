"""
第三步：Embedding + Qdrant 寫入
用 BGE-small-zh embedding，寫入本地 Qdrant（持久化到 ./qdrant_storage）

安裝：
pip install sentence-transformers qdrant-client
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from core.config import (
    BM25_CORPUS_FILE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OUTPUTS_DIR,
    QDRANT_PATH,
    QDRANT_URL,
    create_qdrant_client,
)
from ingest.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_markdown,
)

INDEX_META_FILE = QDRANT_PATH / "index_meta.json"


@dataclass
class BuildState:
    """Whether ingest steps can be skipped."""

    skip_all: bool
    run_convert: bool
    run_index: bool
    message: str
    meta: Optional[Dict[str, Any]] = None


def md_path_for_pdf(pdf_path: Path) -> Path:
    return OUTPUTS_DIR / f"{pdf_path.stem}.md"


def _collection_point_count(client: QdrantClient) -> int:
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    info = client.get_collection(COLLECTION_NAME)
    return int(getattr(info, "points_count", 0) or 0)


def load_index_meta_optional() -> Optional[Dict[str, Any]]:
    if not INDEX_META_FILE.exists():
        return None
    with open(INDEX_META_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def is_index_complete(meta: Optional[Dict[str, Any]] = None) -> bool:
    meta = meta or load_index_meta_optional()
    if not meta or not BM25_CORPUS_FILE.exists():
        return False
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        return False
    try:
        client = create_qdrant_client()
        return _collection_point_count(client) > 0
    except Exception:
        return False


def sources_unchanged(pdf_path: Path, meta: Dict[str, Any], md_path: Path) -> bool:
    """True if PDF has not changed since the indexed sources were produced."""
    pdf_mtime = pdf_path.stat().st_mtime

    recorded = meta.get("source_pdf_mtime")
    if recorded is not None:
        return pdf_mtime <= float(recorded) + 1e-3

    # 舊版 index_meta（無 mtime）：用 MD 是否比 PDF 新來判斷
    for candidate in (md_path, meta.get("source_md")):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and pdf_mtime <= path.stat().st_mtime + 1e-3:
            return True
    return False


def backfill_index_meta_sources(pdf_path: Path, md_path: Path) -> None:
    """補寫 source_pdf_mtime，避免舊索引每次都被迫重建。"""
    meta = load_index_meta_optional()
    if not meta or meta.get("source_pdf_mtime") is not None:
        return
    meta["source_pdf"] = str(pdf_path.resolve())
    meta["source_pdf_mtime"] = pdf_path.stat().st_mtime
    meta["source_md"] = str(md_path.resolve())
    with open(INDEX_META_FILE, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def assess_build_state(pdf_path: Path, *, force: bool = False) -> BuildState:
    pdf_path = pdf_path.resolve()
    md_path = md_path_for_pdf(pdf_path)
    meta = load_index_meta_optional()

    if force:
        return BuildState(
            skip_all=False,
            run_convert=True,
            run_index=True,
            message="--force：將重新轉換並重建索引",
            meta=meta,
        )

    index_ok = is_index_complete(meta)
    unchanged = bool(meta and sources_unchanged(pdf_path, meta, md_path))

    if index_ok and unchanged:
        backfill_index_meta_sources(pdf_path, md_path)
        return BuildState(
            skip_all=True,
            run_convert=False,
            run_index=False,
            message=(
                f"✅ 索引已存在且 PDF 未變更，跳過重建。\n"
                f"   書名: {meta.get('book_title')}\n"
                f"   分塊: {meta.get('num_chunks')} | 模型: {meta.get('embedding_model')}\n"
                f"   建立於: {meta.get('built_at')}\n"
                f"   若要強制重建: uv run python scripts/build_index.py --force"
            ),
            meta=meta,
        )

    md_fresh = md_path.exists() and md_path.stat().st_mtime >= pdf_path.stat().st_mtime - 1e-3
    run_convert = not md_fresh
    run_index = not index_ok or not unchanged

    parts = []
    if run_convert:
        parts.append("PDF→MD")
    if run_index:
        parts.append("建索引")
    message = "將執行：" + "、".join(parts) if parts else "無需變更"

    return BuildState(
        skip_all=False,
        run_convert=run_convert,
        run_index=run_index,
        message=message,
        meta=meta,
    )


def get_md_path() -> Path:
    md_files = sorted(OUTPUTS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError("找不到 outputs/*.md，請先執行 ingest/converter 或 scripts/build_index.py")
    return md_files[0]


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"🔄 載入 Embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ 模型載入完成（向量維度: {model.get_sentence_embedding_dimension()}）")
    return model


def encode_query(model: SentenceTransformer, query: str) -> List[float]:
    """BGE 中文模型：查詢需加 query: 前綴。"""
    return model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()


def embed_chunks(
    model: SentenceTransformer,
    chunks: List[Chunk],
    batch_size: int = 32,
) -> List[List[float]]:
    texts = [chunk.text for chunk in chunks]
    print(f"\n🔄 開始 Embedding {len(texts)} 個分塊...")
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start_time
    print(f"✅ Embedding 完成（{elapsed:.1f}s，{len(texts) / elapsed:.1f} chunks/sec）")
    return embeddings.tolist()


def init_qdrant_client(recreate: bool = False, vector_dim: int = 512) -> QdrantClient:
    if QDRANT_URL:
        client = create_qdrant_client()
        print(f"🔄 連接 Qdrant: {QDRANT_URL}")
    else:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        client = create_qdrant_client()
        print(f"🔄 初始化 Qdrant（持久化: {QDRANT_PATH}）...")

    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"   清除舊集合: {COLLECTION_NAME}")
        except Exception:
            pass
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"✅ 集合建立: {COLLECTION_NAME}（向量維度: {vector_dim}）")
    else:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"集合 {COLLECTION_NAME} 不存在，請先執行: uv run python scripts/build_index.py"
            )

    return client


def ensure_payload_indexes(client: QdrantClient):
    """TEXT index for metadata filter (chapter / heading / book_title)."""
    for field in ("chapter", "heading", "book_title"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.TEXT,
            )
            print(f"   payload index: {field}")
        except Exception as exc:
            print(f"   payload index {field} 略過: {exc}")


def upload_to_qdrant(
    client: QdrantClient,
    chunks: List[Chunk],
    embeddings: List[List[float]],
    book_title: str,
):
    print("\n🔄 上傳到 Qdrant...")
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = {
            "text": chunk.text,
            "book_title": book_title,
            "chapter": chunk.chapter,
            "heading": chunk.heading,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "indexed_at": datetime.now().isoformat(),
        }
        points.append(PointStruct(id=i + 1, vector=embedding, payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ 上傳完成: {len(points)} 個向量")


def save_bm25_corpus(chunks: List[Chunk], book_title: str):
    records = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "chapter": chunk.chapter,
            "heading": chunk.heading,
            "page": chunk.page,
            "book_title": book_title,
            "point_id": index + 1,
        }
        for index, chunk in enumerate(chunks)
    ]
    BM25_CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_FILE, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"💾 BM25 語料: {BM25_CORPUS_FILE} ({len(records)} chunks)")


def save_index_meta(
    book_title: str,
    num_chunks: int,
    vector_dim: int,
    *,
    source_pdf: Path | None = None,
    source_md: Path | None = None,
):
    meta = {
        "book_title": book_title,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "vector_dim": vector_dim,
        "num_chunks": num_chunks,
        "collection_name": COLLECTION_NAME,
        "built_at": datetime.now().isoformat(),
    }
    if source_pdf is not None:
        meta["source_pdf"] = str(source_pdf.resolve())
        meta["source_pdf_mtime"] = source_pdf.stat().st_mtime
    if source_md is not None:
        meta["source_md"] = str(source_md.resolve())
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 索引資訊: {INDEX_META_FILE}")


def load_index_meta() -> Dict[str, Any]:
    if not INDEX_META_FILE.exists():
        raise FileNotFoundError("找不到索引，請先執行 uv run python scripts/build_index.py")
    with open(INDEX_META_FILE, encoding="utf-8") as f:
        return json.load(f)


def hits_to_dicts(results) -> List[Dict]:
    return [
        {
            "text": point.payload.get("text", ""),
            "score": point.score,
            "chapter": point.payload.get("chapter", ""),
            "heading": point.payload.get("heading", ""),
            "chunk_id": point.payload.get("chunk_id"),
        }
        for point in results
    ]


def search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    limit: int = 3,
) -> List[Dict]:
    query_vector = encode_query(model, query)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return hits_to_dicts(response.points)


def print_search_results(query: str, results: List[Dict]):
    print(f"\n❓ 問題: {query}")
    for rank, result in enumerate(results, 1):
        text_preview = result["text"][:80].replace("\n", " ")
        chapter = result.get("chapter") or "(無)"
        heading = result.get("heading") or "(無)"
        print(f"  [{rank}] (相似度: {result['score']:.3f})")
        print(f"       章節: {chapter} → {heading}")
        print(f"       內容: {text_preview}...")


def build_index(
    *,
    force: bool = False,
    pdf_path: Path | None = None,
) -> tuple[QdrantClient, SentenceTransformer, List[Chunk]] | tuple[None, None, None]:
    if pdf_path is not None:
        state = assess_build_state(pdf_path, force=force)
        if state.skip_all:
            print(state.message)
            return None, None, None

    md_path = get_md_path()
    book_title = md_path.stem
    print(f"📖 讀取: {book_title}")

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    print(f"\n⚙️ 生成分塊 (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_OVERLAP})...")
    chunks = chunk_markdown(
        md_text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )
    print(f"✅ 生成 {len(chunks)} 個分塊")
    with_chapter = sum(1 for c in chunks if c.chapter)
    print(f"   含章節標籤的分塊: {with_chapter}/{len(chunks)}")

    model = load_embedding_model(EMBEDDING_MODEL)
    vector_dim = model.get_sentence_embedding_dimension()
    embeddings = embed_chunks(model, chunks)

    print(f"\n🔄 初始化 Qdrant（持久化: {QDRANT_PATH}）...")
    client = init_qdrant_client(recreate=True, vector_dim=vector_dim)
    ensure_payload_indexes(client)
    upload_to_qdrant(client, chunks, embeddings, book_title=book_title)
    save_bm25_corpus(chunks, book_title)
    resolved_pdf = pdf_path.resolve() if pdf_path else None
    if resolved_pdf is None:
        existing = load_index_meta_optional()
        if existing and existing.get("source_pdf"):
            resolved_pdf = Path(existing["source_pdf"])

    save_index_meta(
        book_title,
        len(chunks),
        vector_dim,
        source_pdf=resolved_pdf,
        source_md=md_path,
    )

    return client, model, chunks


if __name__ == "__main__":
    client, model, _ = build_index()

    print("\n🔍 快速搜尋測試:")
    print("=" * 80)
    test_queries = [
        "如何不靠运气致富？",
        "如何找到自己的专长？",
        "幸福是一种可以学习的技能吗？",
    ]
    for query in test_queries:
        results = search(client, model, query)
        print_search_results(query, results)

    print("\n✅ 索引建立完成！")
    print("   接下來: uv run python scripts/chat.py")
    print("   評測預覽: uv run python scripts/eval.py --preview")

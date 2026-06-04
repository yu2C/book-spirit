"""
第三步：Embedding + Qdrant 寫入（多書 incremental）
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from core.collections import LEGACY_COLLECTION, collection_name_for_book, point_count_for_book
from core.config import (
    BM25_CORPUS_FILE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OUTPUTS_DIR,
    QDRANT_PATH,
    QDRANT_URL,
    create_qdrant_client,
)
from ingest.books_registry import (
    STATUS_ARCHIVED,
    STATUS_INDEXED,
    BookEntry,
    ensure_entry_for_pdf,
    get_book,
    load_registry,
    md_path_for_pdf,
    set_book_status,
    slug_from_pdf,
    upsert_registry_entry,
)
from ingest.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_markdown,
)

INDEX_META_FILE = QDRANT_PATH / "index_meta.json"
_POINT_NS = uuid.UUID("a3f2c8e1-4b5d-4e9a-9c7d-1e2f3a4b5c6d")


@dataclass
class BuildState:
    skip_all: bool
    run_convert: bool
    run_index: bool
    message: str
    book_id: str = ""
    meta: Optional[Dict[str, Any]] = None


def bm25_path(book_id: str) -> Path:
    return QDRANT_PATH / f"bm25_{book_id}.json"


def make_point_id(book_id: str, chunk_id: int) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{book_id}:{chunk_id}"))


def load_library_meta_optional() -> Optional[Dict[str, Any]]:
    if not INDEX_META_FILE.exists():
        return None
    with open(INDEX_META_FILE, encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("version") == 2:
        return raw
    # Legacy single-book meta → v2 shape in memory
    book_title = raw.get("book_title", "legacy")
    bid = slug_from_pdf(Path(raw.get("source_pdf", book_title + ".pdf")))
    return {
        "version": 2,
        "embedding_model": raw.get("embedding_model"),
        "collection_name": raw.get("collection_name", COLLECTION_NAME),
        "books": {
            bid: {
                "book_title": book_title,
                "num_chunks": raw.get("num_chunks", 0),
                "source_pdf": raw.get("source_pdf"),
                "source_pdf_mtime": raw.get("source_pdf_mtime"),
                "source_md": raw.get("source_md"),
                "built_at": raw.get("built_at"),
            }
        },
    }


def save_library_meta(books_meta: Dict[str, Dict[str, Any]], vector_dim: int) -> None:
    payload = {
        "version": 2,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "vector_dim": vector_dim,
        "collection_name": COLLECTION_NAME,
        "updated_at": datetime.now().isoformat(),
        "books": books_meta,
    }
    INDEX_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"💾 索引資訊: {INDEX_META_FILE}")


def is_book_indexed(book_id: str) -> bool:
    entry = get_book(book_id)
    if not entry or entry.status != STATUS_INDEXED:
        return False
    if not bm25_path(book_id).exists():
        return False
    lib = load_library_meta_optional()
    if not lib or lib.get("embedding_model") != EMBEDDING_MODEL:
        return False
    if book_id not in (lib.get("books") or {}):
        return False
    try:
        client = create_qdrant_client()
        return point_count_for_book(client, book_id) > 0
    except Exception:
        return False


def sources_unchanged(pdf_path: Path, entry: BookEntry, md_path: Path) -> bool:
    pdf_mtime = pdf_path.stat().st_mtime
    if entry.source_pdf_mtime is not None:
        return pdf_mtime <= float(entry.source_pdf_mtime) + 1e-3
    if md_path.exists():
        return pdf_mtime <= md_path.stat().st_mtime + 1e-3
    return False


def assess_build_state(
    pdf_path: Path,
    book_id: str,
    *,
    force: bool = False,
) -> BuildState:
    pdf_path = pdf_path.resolve()
    md_path = md_path_for_pdf(pdf_path)
    entry = ensure_entry_for_pdf(pdf_path, book_id=book_id)

    if force:
        return BuildState(
            skip_all=False,
            run_convert=True,
            run_index=True,
            message=f"--force：將重新索引 {book_id}",
            book_id=book_id,
        )

    if entry.status == STATUS_ARCHIVED and not force:
        return BuildState(
            skip_all=False,
            run_convert=False,
            run_index=False,
            message=(
                f"書籍 {book_id} 已封存。若要重新全文索引：\n"
                f"   uv run python scripts/build_index.py --book {book_id} --force"
            ),
            book_id=book_id,
        )

    index_ok = is_book_indexed(book_id)
    unchanged = sources_unchanged(pdf_path, entry, md_path)

    if index_ok and unchanged:
        return BuildState(
            skip_all=True,
            run_convert=False,
            run_index=False,
            message=(
                f"✅ [{book_id}] 索引已存在且 PDF 未變更，跳過。\n"
                f"   書名: {entry.book_title}\n"
                f"   分塊: {entry.num_chunks}\n"
                f"   強制重建: uv run python scripts/build_index.py --book {book_id} --force"
            ),
            book_id=book_id,
        )

    md_fresh = md_path.exists() and md_path.stat().st_mtime >= pdf_path.stat().st_mtime - 1e-3
    run_convert = not md_fresh
    run_index = not index_ok or not unchanged
    parts = []
    if run_convert:
        parts.append("PDF→MD")
    if run_index:
        parts.append("建索引")
    message = f"[{book_id}] " + ("將執行：" + "、".join(parts) if parts else "無需變更")

    return BuildState(
        skip_all=False,
        run_convert=run_convert,
        run_index=run_index,
        message=message,
        book_id=book_id,
    )


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"🔄 載入 Embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ 模型載入完成（向量維度: {model.get_sentence_embedding_dimension()}）")
    return model


def encode_query(model: SentenceTransformer, query: str) -> List[float]:
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


def ensure_collection(client: QdrantClient, book_id: str, vector_dim: int) -> str:
    if QDRANT_URL:
        print(f"🔄 連接 Qdrant: {QDRANT_URL}")
    else:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    cname = collection_name_for_book(book_id)
    if not client.collection_exists(cname):
        client.create_collection(
            collection_name=cname,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"✅ 建立集合: {cname}（維度 {vector_dim}）")
    return cname


def ensure_payload_indexes(client: QdrantClient, collection_name: str):
    for field, schema in (
        ("chapter", PayloadSchemaType.TEXT),
        ("heading", PayloadSchemaType.TEXT),
        ("book_title", PayloadSchemaType.TEXT),
        ("book_id", PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
            print(f"   payload index: {field}")
        except Exception as exc:
            print(f"   payload index {field} 略過: {exc}")


def delete_vectors_for_book(client: QdrantClient, book_id: str) -> None:
    cname = collection_name_for_book(book_id)
    try:
        if client.collection_exists(cname):
            client.delete_collection(cname)
            print(f"   已刪除集合: {cname}")
    except Exception as exc:
        print(f"   刪除集合略過: {exc}")
    if client.collection_exists(LEGACY_COLLECTION):
        try:
            client.delete(
                collection_name=LEGACY_COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
                ),
            )
            print(f"   已自 {LEGACY_COLLECTION} 移除 book_id={book_id}")
        except Exception:
            pass


def upload_to_qdrant(
    client: QdrantClient,
    chunks: List[Chunk],
    embeddings: List[List[float]],
    *,
    collection_name: str,
    book_id: str,
    book_title: str,
):
    print(f"\n🔄 上傳到 Qdrant（{collection_name}）...")
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        pid = make_point_id(book_id, chunk.chunk_id)
        payload = {
            "text": chunk.text,
            "book_id": book_id,
            "book_title": book_title,
            "chapter": chunk.chapter,
            "heading": chunk.heading,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "indexed_at": datetime.now().isoformat(),
        }
        points.append(PointStruct(id=pid, vector=embedding, payload=payload))
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ 上傳完成: {len(points)} 個向量（{book_id}）")


def save_bm25_corpus(chunks: List[Chunk], *, book_id: str, book_title: str):
    path = bm25_path(book_id)
    records = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "chapter": chunk.chapter,
            "heading": chunk.heading,
            "page": chunk.page,
            "book_id": book_id,
            "book_title": book_title,
            "point_id": make_point_id(book_id, chunk.chunk_id),
        }
        for chunk in chunks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"💾 BM25: {path} ({len(records)} chunks)")


def build_index_for_book(
    book_id: str,
    pdf_path: Path,
    *,
    force: bool = False,
) -> bool:
    """Index one book. Returns True if indexing ran."""
    state = assess_build_state(pdf_path, book_id, force=force)
    if state.skip_all:
        print(state.message)
        return False
    if state.run_index is False and not state.run_convert:
        print(state.message)
        return False

    entry = ensure_entry_for_pdf(pdf_path, book_id=book_id)
    md_path = Path(entry.md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"找不到 {md_path}，請先轉換 PDF")

    book_title = entry.book_title
    print(f"📖 [{book_id}] {book_title}")

    with open(md_path, encoding="utf-8") as handle:
        md_text = handle.read()

    print(f"\n⚙️ 生成分塊 (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_OVERLAP})...")
    chunks = chunk_markdown(md_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
    print(f"✅ 生成 {len(chunks)} 個分塊")
    with_chapter = sum(1 for c in chunks if c.chapter)
    print(f"   含章節標籤: {with_chapter}/{len(chunks)}")

    model = load_embedding_model(EMBEDDING_MODEL)
    vector_dim = model.get_sentence_embedding_dimension()
    embeddings = embed_chunks(model, chunks)

    client = create_qdrant_client()
    delete_vectors_for_book(client, book_id)
    cname = ensure_collection(client, book_id, vector_dim)
    ensure_payload_indexes(client, cname)
    upload_to_qdrant(
        client,
        chunks,
        embeddings,
        collection_name=cname,
        book_id=book_id,
        book_title=book_title,
    )
    save_bm25_corpus(chunks, book_id=book_id, book_title=book_title)

    lib = load_library_meta_optional() or {"books": {}}
    books_meta = dict(lib.get("books") or {})
    books_meta[book_id] = {
        "book_title": book_title,
        "num_chunks": len(chunks),
        "qdrant_collection": cname,
        "source_pdf": str(pdf_path.resolve()),
        "source_pdf_mtime": pdf_path.stat().st_mtime,
        "source_md": str(md_path.resolve()),
        "built_at": datetime.now().isoformat(),
    }
    save_library_meta(books_meta, vector_dim)

    entry.status = STATUS_INDEXED
    entry.num_chunks = len(chunks)
    entry.source_pdf_mtime = pdf_path.stat().st_mtime
    entry.indexed_at = datetime.now().isoformat()
    entry.md_path = str(md_path)
    upsert_registry_entry(entry)
    return True


def archive_book(book_id: str) -> None:
    entry = get_book(book_id)
    if not entry:
        raise KeyError(f"未知 book_id: {book_id}")

    client = create_qdrant_client()
    delete_vectors_for_book(client, book_id)

    path = bm25_path(book_id)
    if path.exists():
        path.unlink()
        print(f"   已刪除 {path.name}")

    set_book_status(book_id, STATUS_ARCHIVED, num_chunks=0)
    lib = load_library_meta_optional()
    if lib and book_id in (lib.get("books") or {}):
        books_meta = dict(lib["books"])
        books_meta.pop(book_id, None)
        save_library_meta(books_meta, lib.get("vector_dim", 512))
    print(f"✅ [{book_id}] 已封存（僅保留 SQLite 筆記，問答走 Memory）")


def list_indexed_book_ids() -> List[str]:
    reg = load_registry()
    return [bid for bid, e in reg.items() if e.status == STATUS_INDEXED]


# --- Legacy helpers (eval / single-md scripts) ---


def load_index_meta() -> Dict[str, Any]:
    meta = load_library_meta_optional()
    if not meta:
        raise FileNotFoundError("找不到索引，請先執行 scripts/build_index.py")
    return meta


def get_md_path() -> Path:
    md_files = sorted(OUTPUTS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError("找不到 outputs/*.md")
    return md_files[0]


def build_index(*, force: bool = False, pdf_path: Path | None = None):
    """Backward-compatible: index one PDF."""
    if pdf_path is None:
        from core.config import SAMPLE_BOOKS_DIR

        pdfs = sorted(SAMPLE_BOOKS_DIR.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("sample_books 內無 PDF")
        pdf_path = pdfs[0]
    book_id = slug_from_pdf(pdf_path)
    build_index_for_book(book_id, pdf_path, force=force)
    return create_qdrant_client(), load_embedding_model(), None


def init_qdrant_client(
    recreate: bool = False,
    vector_dim: int = 512,
    book_id: str | None = None,
) -> QdrantClient:
    """Legacy entry for eval scripts."""
    client = create_qdrant_client()
    bids = list_indexed_book_ids()
    bid = book_id or (bids[0] if bids else "default")
    if recreate:
        delete_vectors_for_book(client, bid)
    ensure_collection(client, bid, vector_dim)
    return client


def is_index_complete(meta: Optional[Dict[str, Any]] = None) -> bool:
    """Any book indexed (legacy API)."""
    lib = meta or load_library_meta_optional()
    if not lib:
        return False
    for bid in (lib.get("books") or {}):
        if is_book_indexed(bid):
            return True
    if BM25_CORPUS_FILE.exists() and lib.get("embedding_model") == EMBEDDING_MODEL:
        return True
    return False

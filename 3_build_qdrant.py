"""
第三步：Embedding + Qdrant 寫入
用 BGE-small-zh embedding，寫入本地 Qdrant（持久化到 ./qdrant_storage）

安裝：
pip install sentence-transformers qdrant-client
"""

import importlib.util
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType

from rag_config import BM25_CORPUS_FILE, QDRANT_URL, create_qdrant_client

# 從步驟 2 共用分塊邏輯
_chunk_spec = importlib.util.spec_from_file_location(
    "chunk_embed",
    Path(__file__).parent / "2_chunk_and_embed.py",
)
_chunk_module = importlib.util.module_from_spec(_chunk_spec)
_chunk_spec.loader.exec_module(_chunk_module)

Chunk = _chunk_module.Chunk
chunk_markdown = _chunk_module.chunk_markdown
DEFAULT_CHUNK_SIZE = _chunk_module.DEFAULT_CHUNK_SIZE
DEFAULT_OVERLAP = _chunk_module.DEFAULT_OVERLAP

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "books"
QDRANT_PATH = Path(__file__).parent / "qdrant_storage"
INDEX_META_FILE = QDRANT_PATH / "index_meta.json"


def get_md_path() -> Path:
    md_files = sorted(Path("outputs").glob("*.md"))
    if not md_files:
        raise FileNotFoundError("找不到 outputs/*.md，請先執行 1_convert_pdf_to_md.py")
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
                f"集合 {COLLECTION_NAME} 不存在，請先執行: python 3_build_qdrant.py"
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
    print(f"\n🔄 上傳到 Qdrant...")
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


def save_index_meta(book_title: str, num_chunks: int, vector_dim: int):
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
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 索引資訊: {INDEX_META_FILE}")


def load_index_meta() -> Dict[str, Any]:
    if not INDEX_META_FILE.exists():
        raise FileNotFoundError("找不到索引，請先執行 python 3_build_qdrant.py")
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


def build_index():
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
    save_index_meta(book_title, len(chunks), vector_dim)

    return client, model, chunks


if __name__ == "__main__":
    client, model, _ = build_index()

    print(f"\n🔍 快速搜尋測試:")
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
    print("   接下來執行: python 4_test_search_quality.py")
    print("   或預覽模式: python 4_test_search_quality.py --preview")

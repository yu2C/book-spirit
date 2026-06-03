# ETL 入庫流程

與 [`ARCHITECTURE.md`](../ARCHITECTURE.md) 的 **ingest** 層對應。

| 階段 | 模組 | 說明 |
|------|------|------|
| **Extract** | `ingest/converter.py` | PDF → Markdown（MarkItDown） |
| **Transform** | `ingest/chunker.py` | 分塊、章節 metadata |
| **Load** | `ingest/indexer.py` | BGE embedding + Qdrant + BM25 語料 |

| 下游 | 入口 | 說明 |
|------|------|------|
| 問答 | `scripts/chat.py` | 互動問答 + `/save` |
| 評測 | `scripts/eval.py` | 檢索品質（不含 LLM） |
| API | `python -m api` | FastAPI |

## 一鍵執行

```bash
bash etl/run_pipeline.sh
```

等同：

```bash
uv run python scripts/build_index.py
uv run python scripts/eval.py --preview
```

## Docker

索引需在 host 或容器外建立後掛載 `qdrant_storage`，或：

```bash
QDRANT_URL=http://localhost:6333 uv run python scripts/build_index.py
docker compose up -d
```

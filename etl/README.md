# ETL — Extract, Transform, Load

書籍入庫流程：**Extract PDF → Transform chunk → Load Qdrant**

## 流程對照

| ETL 階段 | 腳本 | 說明 |
|----------|------|------|
| **Extract** | `1_convert_pdf_to_md.py` | PDF → Markdown（MarkItDown） |
| **Transform** | `2_chunk_and_embed.py` | 分塊實驗、章節 metadata 提取 |
| **Load** | `3_build_qdrant.py` | BGE embedding + 寫入 Qdrant |

評測與服務：

| 步驟 | 腳本 | 說明 |
|------|------|------|
| Eval | `4_test_search_quality.py` | 檢索品質評測（不含 LLM） |
| QA CLI | `5_generate_answer_with_llm.py` | 本地 Ollama 問答 |
| API | `6_fastapi_server.py` | FastAPI `/search` + `/ask` |

## 一鍵執行

```bash
# 從 repo 根目錄
bash etl/run_pipeline.sh
```

或逐步：

```bash
python 1_convert_pdf_to_md.py
python 3_build_qdrant.py          # 已內建步驟 2 的分塊邏輯
python 4_test_search_quality.py --preview
```

## Docker 部署前

1. 在本機或 CI 跑完 ETL，讓 Qdrant 有 `books` collection  
2. 若用 `docker compose`，需將索引載入 Qdrant 容器（重建 `3_build_qdrant.py` 並指向 `QDRANT_URL=http://localhost:6333`）

```bash
QDRANT_URL=http://localhost:6333 python 3_build_qdrant.py
docker compose up -d
curl http://localhost:8000/health
```

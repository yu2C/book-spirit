# book-spirit Todo

> 目標：補齊 104 AI/RAG 職缺常見關鍵字（LangChain、CI/CD、Cloud、SQL、測試、上線敘事）。  
> 用法：複製到 [book-spirit](https://github.com/yu2C/book-spirit) repo 根目錄的 `Todo.md`，完成一項就勾選。

---

## Phase 0 — 文件與對外敘事（1–2 天）

- [x] **README 架構圖**：PDF → Markdown → chunk → embed → Qdrant → FastAPI →（可選）LLM，附一張 mermaid 或 PNG
- [x] **README「求職向摘要」**：3–5 行英文 + 中文，列出 RAG / Vector DB / FastAPI / eval
- [x] **關鍵字對照表**（README 一節）：自研 pipeline vs LangChain 可互換；Qdrant = VectorDB
- [x] **Prompt / 檢索評測**：在 README 說明 `4_test_search_quality.py` 量什麼（precision、相似度門檻、bad case）

---

## Phase 1 — HR 搜尋關鍵字（3–5 天）

- [x] **LangChain 整合（薄層即可）**：新增 `langchain_retriever.py`，用 LangChain 接 Qdrant/BGE，API 可切換 `native` / `langchain`
- [x] **（擇一）LlamaIndex**：README 寫清「可接 LlamaIndex」並留擴充說明
- [x] **Agent 最小 demo（可選）**：LangGraph `retrieve → generate` workflow（`rag_graph.py`）
- [x] **requirements.txt 分層**：`requirements.txt`（核心）+ `requirements-dev.txt` + `requirements-langchain.txt`

---

## Phase 2 — 後端 / SQL / ETL 敘事（3–5 天）

- [x] **PostgreSQL query log**：FastAPI 將每次 query、top-k 結果 id、latency 寫入 Postgres（Docker Compose 起 DB）
- [x] **docker-compose.yml**：`qdrant` + `postgres` + `api` 一鍵起
- [x] **ETL 腳本命名**：`etl/README.md` + `etl/run_pipeline.sh`
- [x] **REST API 文件**：README 補 `/health`、`/search`、`/ask`；Swagger 見 `/docs`

---

## Phase 3 — 測試與 CI/CD（2–4 天）

- [x] **pytest 煙霧測試**：chunk 函式、embed mock、search 回傳格式（不需 GPU）
- [x] **GitHub Actions `ci.yml`**：`pip install` → `pytest` → ruff
- [x] **CI badge**：README 頂部 build status
- [x] **pre-commit（可選）**：`.pre-commit-config.yaml` — ruff format / ruff check

---

## Phase 4 — 上線 / Cloud（暫緩，詳見根目錄 `Todo.md` Phase 4）

> 建議：**Render** + **Qdrant Cloud** + Postgres；雲端只展示 `/health` + `/search`，Ollama 留本地。

- [x] **容器化 API**：`Dockerfile`（已完成）
- [ ] **4.1–4.6 上雲步驟**：Qdrant Cloud → Render PG → Render Web Service → 公開 URL → README
- [x] **環境變數範本**：`.env.example`
- [x] **簡易 monitoring**：`/health` 回 Qdrant + Postgres 連線狀態

---

## Phase 4b — 檢索進階（Hybrid + Rerank + Metadata Filter）

> 細項步驟見根目錄 `Todo.md` Phase 4b。順序：Metadata filter → Hybrid BM25+RRF → Reranker。

- [ ] **Metadata filter**：Qdrant payload filter（chapter / heading）
- [ ] **Hybrid search**：BM25 + 向量 + RRF
- [ ] **Reranker**：BGE reranker cross-encoder
- [ ] **Eval 對照**：`4_test_search_quality.py` before/after

---

## Phase 5 — LLM 回答層（若尚未完成，5–7 天）

- [ ] **RAG answer endpoint**：retrieve → prompt template → LLM（Ollama 本地或 OpenAI API 二選一）
- [ ] **Prompt template 版本化**：`prompts/` 目錄 + 簡短說明調參邏輯
- [ ] **回答品質 eval（輕量）**：10 題固定 Q&A，人工或 regex 檢查 citation 是否來自 top-k
- [ ] **串流（可選）**：FastAPI SSE 給前端或 curl demo

---

## Phase 6 — 履歷 / 104 同步（完成 Phase 1–4 後）

- [ ] **cv.md bullet 更新**：LangChain（optional）、Postgres log、GitHub Actions、cloud URL
- [ ] **104 專長標籤**：Python、RAG、LangChain、FastAPI、Docker、Qdrant、CI/CD、Linux
- [ ] **GitHub About / topics**：`rag`, `langchain`, `qdrant`, `fastapi`, `bge`, `retrieval-evaluation`

---

## 不做 / 低優先（避免 scope 爆炸）

- [ ] ~~重寫成 NestJS / React 全端~~（百應向；除非確定要投全端）
- [ ] ~~換 OpenSearch~~（華碩 JD 有提，但 Qdrant 已足夠展示 VectorDB）
- [ ] ~~微調 LLM~~（JD 加分即可；有 SchNet/FHE 可面試再講）

---

## 建議執行順序（最小可展示）

1. Phase 0 README  
2. Phase 1 LangChain 薄層  
3. Phase 3 pytest + GitHub Actions  
4. Phase 2 docker-compose + Postgres log  
5. Phase 4 部署一個 public health URL  

完成 **0 + 1 + 3 + 4** 即可大幅強化華碩 / 威剛 / 捷鵬（自動化敘事在 JAM/CV）相關敘事。

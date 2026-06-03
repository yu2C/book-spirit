# RAG 修復 Todo

- [x] 1. 修復 Qdrant 向量維度（384 → 依模型動態設定）
- [x] 2. 更新測試問題為《納瓦爾寶典》主題
- [x] 3. 步驟 3 共用步驟 2 的分塊邏輯（含 overlap）
- [x] 4. 支援無 `#` 標題的 MD 章節偵測

## 搜尋品質工具修復

- [x] 3_build_qdrant.py 改為持久化儲存 + BGE query 前綴
- [x] 4_test_search_quality.py 實際連接 Qdrant 搜尋評估
- [x] 測試問題改為《納瓦爾寶典》
- [x] 修正錯誤的 heading 標籤（∨、[78] 碎片）

## LangChain / LangGraph 整合

- [x] rag_config.py / rag_native.py 共用設定與 native pipeline
- [x] langchain_retriever.py — LangChain Qdrant + BGE retriever
- [x] rag_graph.py — LangGraph retrieve → generate
- [x] 6_fastapi_server.py — backend 切換 native / langchain / langgraph
- [x] requirements 分層（requirements-langchain.txt）
- [x] README Phase 0（架構圖、求職摘要、關鍵字表、評測說明）

## Phase 2 — Docker / Postgres / ETL

- [x] query_log.py + Postgres query_logs 表
- [x] docker-compose.yml（qdrant + postgres + api）+ Dockerfile
- [x] etl/README.md + run_pipeline.sh
- [x] /search endpoint + /health 含 postgres 狀態
- [x] .env.example

## Phase 3 — 測試與 CI

- [x] tests/ pytest 煙霧測試（12 tests）
- [x] .github/workflows/ci.yml + README CI badge
- [x] .pre-commit-config.yaml + requirements-test.txt

---

## Phase 4 — 上雲（暫緩，先規劃不實作）

> **建議平台：Render**（第一次上雲最直覺）。策略：雲端只公開 `/health` + `/search` + Postgres log；LLM 留本地 Ollama 或之後接 API。

- [ ] **4.1 Qdrant Cloud**：開 free cluster → 本機 `QDRANT_URL=... python 3_build_qdrant.py` 灌索引
- [ ] **4.2 Render PostgreSQL**：建立 DB → 設定 `DATABASE_URL`（沿用 `query_logs`）
- [ ] **4.3 Render Web Service**：連 GitHub repo，用現有 `Dockerfile` deploy API
- [ ] **4.4 環境變數**：`QDRANT_URL`、`DATABASE_URL`、`RAG_BACKEND=native`；**不**在雲端跑 Ollama
- [ ] **4.5 驗證**：公開 `https://xxx/health` + `POST /search` curl 通
- [ ] **4.6 README**：補 Render deploy 步驟 + demo URL（health 或 search 即可）
- [x] Dockerfile / docker-compose / `.env.example`（本地已備好）

**不做（現階段）：** 雲端 GPU Ollama、自架 Qdrant 容器（除非 Qdrant Cloud 不夠用）

---

## Phase 4b — 檢索進階（Hybrid + Rerank + Metadata Filter）

> **目標：** 提升 `4_test_search_quality.py` precision@3，API 可選開關，不破壞現有 native / langchain backend。  
> **建議順序：** Metadata filter → Hybrid → Reranker（由易到難、每步都可獨立 eval）。

### 架構（規劃）

```
query + optional filters (chapter/heading)
    ↓
[Metadata filter] — Qdrant payload filter（窄化候選集）
    ↓
[Hybrid retrieve] — 向量 top-N ∥ BM25 top-N → RRF 合併 → 候選 M 筆
    ↓
[Reranker] — cross-encoder 重排 → 最終 top_k
    ↓
/search、/ask
```

| 元件 | 技術選型 | 備註 |
|------|----------|------|
| Metadata filter | Qdrant `Filter` + 現有 payload（chapter, heading, book_title） | 不增模型 |
| BM25 | `rank_bm25` 或 `bm25s`；ETL 時存 `outputs/bm25_corpus.json` | 單書 ~數百 chunk，記憶體可接受 |
| 融合 | RRF（Reciprocal Rank Fusion），不用手調權重 | k=60 常見預設 |
| Reranker | `BAAI/bge-reranker-base`（中文、CPU 可跑） | 候選 15–20 → 輸出 top_k=3 |
| Eval | 擴充 `4_test_search_quality.py` 比較 before/after | 同一 10 題固定集 |

### Step 1 — Metadata filter（約 0.5–1 天）

- [x] **1.1** `rag_config.py`：定義 `SearchFilters`（chapter / heading / book_title 可選）
- [x] **1.2** `rag_native.retrieve()`：Qdrant `query_points(..., query_filter=...)` 支援子字串 match
- [x] **1.3** `6_fastapi_server.py`：`SearchRequest` / `AskRequest` 加 optional `chapter`、`heading`
- [x] **1.4** 測試：pytest mock Qdrant filter 參數；手動測「第一部分 + 專長」類問題
- [x] **1.5** README：說明 filter 用法與限制（heading 為 PDF 解析結果，可能不完整）

### Step 2 — Hybrid BM25 + 向量（約 1–2 天）

- [x] **2.1** ETL：`3_build_qdrant.py` 或新腳本輸出 BM25 語料（chunk_id, text, tokenized 可選）
- [x] **2.2** 新模組 `rag_hybrid.py`：`bm25_search()` + `vector_search()` + `rrf_merge()`
- [x] **2.3** 常數：`RETRIEVE_CANDIDATES=20`（hybrid 後）→ 再交 reranker 或直出 top_k
- [x] **2.4** `NativeRAG.retrieve(mode="vector"|"hybrid")` 或 env `RETRIEVAL_MODE=hybrid`
- [x] **2.5** `4_test_search_quality.py`：加 `--mode hybrid` 對照 precision
- [x] **2.6** pytest：RRF 合併邏輯 unit test（假資料，不需 GPU）

### Step 3 — Reranker（約 1 天）

- [x] **3.1** 新模組 `rag_rerank.py`：`Reranker` 包 `CrossEncoder('BAAI/bge-reranker-base')`
- [x] **3.2** 流程：hybrid 取 M=15 → rerank → top_k；API 加 `use_rerank: bool = True`
- [x] **3.3** 回傳 score 區分：`vector_score` / `rerank_score`（或統一 `score` 並在 payload 標來源）
- [x] **3.4** LangChain backend：可選接 `ContextualCompressionRetriever` 或 native 共用 rerank 函式
- [x] **3.5** eval + README：記錄 M4/WSL rerank 延遲（通常 +100–300ms CPU）

### Step 4 — 整合與收斂（約 0.5 天）

- [x] **4.1** 預設策略：`vector`（現狀）／`hybrid`／`hybrid+rerank` 三檔，環境變數或 query param
- [x] **4.2** `query_log` 多記 `retrieval_mode`、`filters`、`rerank` 欄位
- [x] **4.3** 更新 README 架構圖 + 求職關鍵字（Hybrid Search、Reranker、Metadata Filtering）
- [x] **4.4** 更新 `requirements.txt`：`rank-bm25`（或 bm25s）、reranker 隨 sentence-transformers 已有 CrossEncoder

### 刻意不做（本專案）

- [ ] ~~Semantic cache / Redis~~
- [ ] ~~User 角色 / multi-tenant~~
- [ ] ~~Qdrant sparse vector 雙寫~~（BM25 sidecar 對單書已足夠；以後多書再考慮）

### 驗收標準

- [ ] `4_test_search_quality.py` hybrid+rerank 平均 precision@3 **≥ 現有 vector-only**（或 bad case 明顯減少）
- [ ] `/search` 帶 `chapter` filter 時，top-3 章節標籤一致率提高
- [x] CI pytest 仍全過（rerank / hybrid 在 CI 用 mock，不載入 CrossEncoder）

# 技術筆記

個人學習 RAG 時的複習用：名詞、路徑、踩坑。操作見 [README.md](../README.md)，模組見 [ARCHITECTURE.md](../ARCHITECTURE.md)。

---

## 名詞表

| 詞 | 全名 / 展開 | 在本專案裡 |
|----|-------------|------------|
| **RAG** | Retrieval-Augmented Generation（檢索增強生成） | 先從書裡搜段落，再交 LLM 根據段落回答 |
| **ETL** | Extract–Transform–Load | 入庫三步：轉 MD → 分塊 → 寫 Qdrant + BM25（`build_index.py`） |
| **BGE** | BAAI General Embedding（如 `bge-small-zh-v1.5`） | 把文字變向量；**問句** embed 前加 `query:`，書中 chunk 不加 |
| **Embedding** | 嵌入向量 | 語意相近的段落，向量距離較近 |
| **Chunk** | 分塊 | 可檢索的小段；預設 `semantic` 切，上限 512 字、overlap 64 |
| **Vector 檢索** | 向量 / 語意檢索 | BGE + Qdrant 找最近鄰 |
| **BM25** | Best Matching 25 | 關鍵字匹配；語料在 `qdrant_storage/bm25_<book_id>.json` |
| **Hybrid** | 混合檢索 | 向量一路 + BM25 一路，再合併 |
| **RRF** | Reciprocal Rank Fusion（倒數排名融合） | 合併兩份排名，不硬湊分數尺度（`RRF_K=60`） |
| **Rerank** | 重排序 | cross-encoder 對 **(問題, 段落)** 打分精排；候選約 15 → 留 top_k |
| **top_k** | — | 最後送進 LLM 的 **段落個數**（預設 5），不是 chunker 切幾刀 |
| **Qdrant** | — | 本地向量庫，`qdrant_storage/`；多書時 `book_<book_id>` collection |
| **Metadata** | 中繼資料 | payload：`book_id`、`book_title`、`chapter`、`heading`、`page`、`chunk_id` |
| **[來源 N]** | — | `build_prompt` 替檢索段落編號；SYSTEM_PROMPT 要求回答標同號 |
| **Memory** | — | SQLite 筆記（`/save`），**不是** Qdrant 裡的書正文 |
| **prepare_ask** | — | 檢索**前**：書籍 scope、SQLite 筆記、profile、可選 planner |
| **build_prompt** | — | 檢索**後**：top_k 段落 + 筆記 + profile 組成送 LLM 的字串 |
| **Fallback** | 降級梯子 | 低分時：hybrid_rerank → 加大 top_k → hybrid → vector |
| **Golden eval** | 金標檢索測 | `eval.py --golden` + `test_cases.json`，**不跑 LLM** |

---

## 向量資料庫簡介

一般資料庫用 **精確匹配**（`id = 5`、`title LIKE '%專長%'`）。  
**向量資料庫**存的是 **embedding 向量**，查詢時找「和問句向量 **最接近**」的幾筆——適合 **換句話說、語意相近** 的搜尋，不必字面一樣。

### 和本專案的對應

| 概念 | 是什麼 | 本 repo |
|------|--------|---------|
| **Collection** | 一組向量的「櫃子」 | 每本書一個：`book_<book_id>`（見 `core/collections.py`） |
| **Point** | 櫃子裡的一格 = 一筆資料 | 對應 **一個 chunk** |
| **Vector** | 該 chunk 的 BGE 數字陣列 | 建索引時 `indexer.py` 寫入 |
| **Payload** | 不必算距離也能帶回的文字與標籤 | `text`、`chunk_id`、`chapter`、`heading`、`book_id`… |
| **Point ID** | 資料庫主鍵（UUID 字串） | `make_point_id(book_id, chunk_id)` → 同書同塊 **重建可覆寫** |

### `chunk_id` vs point

- **`chunk_id`**：切書時編的整數（0, 1, 2…），只在 **同一本書** 內有意義。  
- **Point**：Qdrant 裡那一筆；用 `uuid5(namespace, f"{book_id}:{chunk_id}")` 綁死，避免重建索引時重複堆疊。

問答時：問題 → BGE（加 `query:`）→ 在 collection 裡 **近似最近鄰（ANN）** → 取 top 候選 → 再 hybrid / rerank。

### 和 BM25 檔的分工

| | 向量庫（Qdrant） | BM25 檔（`bm25_*.json`） |
|--|------------------|---------------------------|
| 擅長 | 語意像不像 | 關鍵字、專名對不對 |
| 本專案 | 主檢索 | hybrid 時與向量 **RRF 合併** |

向量庫 **不存** `/save` 筆記；筆記在 SQLite，關鍵字另路進 prompt。

### 為何選 Qdrant（向量庫產品）

| 產品 | 特點 |
|------|------|
| **Qdrant** | 開源、本地目錄、payload filter；本專案預設 |
| Chroma | 更簡、嵌入式 |
| FAISS | 快，但要自己管 metadata / 持久化 |
| pgvector | 向量和業務同一 Postgres；本專案 Postgres 只 optional 記 query log |

單書幾百～幾千 chunk、個人機器：本地 Qdrant 檔案 `qdrant_storage/` 夠用；刪了可 `build_index` 重建。

---

## 問答路徑（四步）

```
chat.py / API
  → prepare_ask        # 哪本書、SQLite 筆記、profile、planner（尚無 Qdrant 段落）
  → retrieve           # hybrid：BGE 向量(Qdrant) + BM25 → RRF；再 rerank；必要時 fallback
  → build_prompt       # [來源 1]… + 筆記 + profile + 問題
  → Ollama 生成
```

LangGraph 只是把上述拆成圖節點（`integrations/langgraph.py`），邏輯共用 `core/ask_orchestrator.py`。

---

## 兩條「記憶」路線

| | 書裡段落 | `/save` 筆記 |
|--|----------|----------------|
| 存在 | Qdrant | SQLite `data/reading_memory.db` |
| 怎麼找 | BGE + hybrid + rerank | 關鍵字 `search_relevant` |
| 進 prompt | 「提供的文本內容」 | 「讀者過往筆記」等區塊 |
| 換 embedding 重建索引 | 要重 embed | **不用動** |

---

## 多書：`book_id` / `all` / 封存

| 設定 | 檢索 | 筆記 |
|------|------|------|
| 預設或 `/book naval-almanac` | 只搜該書 collection | 可帶該書相關筆記 |
| `/book all` | 跨所有已索引書 | 筆記不限單書 |
| **封存** `build_index.py --archive <id>` | **不檢索原文**（`rag_enabled=false`） | 仍可用 SQLite 筆記回答 |

封存後向量與 BM25 會刪，registry 標 `archived`；`resolve_ask_context` 會提示「僅使用筆記」。

---

## Golden eval 怎麼算 pass？

**不是**和標準答案算 0.75 相似度，也**不是**測 LLM。

規則（`eval/golden.py`）：

- 每題有 `must_contain_any`：一串中文關鍵句
- **top-k 任一段落**只要包含其中**任一句** → pass
- 另可選看 `expected_chapter` 是否對上

不測 LLM：生成有隨機性，eval 只鎖定 **檢索有沒有撈到對的 chunk**。

人工互動評測裡的「相似度 > 0.75」是**看 BGE 分數的經驗值**，和金標自動 pass 是兩回事。

---

## Overview 類問題（「這本書在講什麼」）

RAG 只有 top-k **片段**，很難代表整本書；常檢到致謝、目錄等垃圾段。

本專案：

- `empty_retrieval_answer` / `is_overview_question` 會給較明確的「請改問具體章節」提示
- `retrieval_top_k` 對 overview 類可能加大 k
- 可開 `USE_QUERY_PLANNER=true` 改寫檢索句（多一次 Ollama）

**沒做：** 專門為 overview 再跑一輪完全不同的 re-prompt 檢索。

---

## 檢索信心低時

| 機制 | 做什麼 |
|------|--------|
| **fallback 梯子** | 換策略、加大 top_k，仍低分才降級 |
| **low_confidence 提示** | 回答前提醒對照引用、可換問法 |
| **ingest_hint** | MarkItDown PDF 且分低時，建議 MinerU 重建 |
| **`/debug`** | 看每輪 strategy、top1 分數 |

不是「同一套流程無限 re-prompt」；低分時主要是 **降級檢索 + 提示使用者**。

---

## RRF（公式直覺）

Hybrid 時向量與 BM25 各有一份排名。對段落 \(d\) 在第 \(r\) 名（0 起算）加 \(\frac{1}{k+r+1}\)。  
兩邊都靠前的段落總分最高。

---

## Chunker

| 模式 | 適合 |
|------|------|
| `semantic`（預設） | 短句密集（納瓦尔：句級相似度找切點） |
| `native` | 長篇章節敘事（字數切 + 章節 regex） |

---

## PDF：MarkItDown vs MinerU

| | MarkItDown | MinerU |
|--|------------|--------|
| 輸出 | 輕量 MD，常缺 `#` 標題 | 版面、表格、標題 often 較完整 |
| 成本 | 快、依賴少 | 慢、吃 CPU/RAM、要額外 group |
| 格式 | PDF、DOCX、EPUB… | 本專案 mainly 難 PDF |

表格多、掃描版、章節亂 → 值得試 MinerU + `eval.py --golden` 對照。

---

## Native vs LangChain / LangGraph

主幹 `NativeRAG`；LangGraph 為圖編排對照；LangChain 薄包裝。學習重點是 RAG 零件，不是 Agent 框架。

---

## 常見誤解

| 以為 | 實際 |
|------|------|
| prepare 時就有 Qdrant 段落 | retrieve 完才 `build_prompt` |
| BGE、vector、RRF 三條平行線 | BGE 驅動 vector；RRF 在 hybrid 裡合併排名 |
| rerank 用「問句和答案」 | 還沒生成；是 **(問題, chunk)** |
| golden = 答案相似度 0.75 | top-k 含 `must_contain_any` 任一句即 pass |
| 筆記也在 Qdrant | SQLite，關鍵字進 prompt |
| chunk_id 就是 Qdrant 主鍵 | chunk_id 是書內編號；point id 是 UUID，兩者經 `make_point_id` 對應 |

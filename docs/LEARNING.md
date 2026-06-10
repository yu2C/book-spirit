# 技術筆記

個人複習用：面試可講的取捨與踩坑。操作步驟見 [README.md](../README.md)。

---

## 名詞

| 詞 | 一句話 |
|----|--------|
| Chunk | 可檢索的小段（預設 semantic 切，上限 512 字） |
| BM25 | 關鍵字匹配，專名/術語 often 更穩 |
| RRF | 合併向量排名與 BM25 排名（`k=60`，見 `core/hybrid.py`） |
| Rerank | cross-encoder 對候選精排，更慢更準 |
| Memory | SQLite 筆記，**不是** Qdrant 裡的書段落 |

---

## RRF（Reciprocal Rank Fusion）

Hybrid 時向量與 BM25 各有一份排名。對段落 \(d\) 在第 \(r\) 名（0 起算）加 \(\frac{1}{k+r+1}\)。  
兩邊都靠前的段落總分最高，不必把 BM25 分數和 cosine 硬湊同一尺度。

---

## 為何用 Qdrant？

| 考量 | 說明 |
|------|------|
| 本地檔案 | `qdrant_storage/`，刪了可重建，符合 toy 迭代 |
| Payload filter | `book_id`、`chapter` 子字串過濾 |
| 規模 | 單書幾百～幾千 chunk，不需分散式 |

Chroma 更簡、FAISS 偏函式庫需自管 metadata；本專案選 Qdrant 練 API 與 filter。

---

## Native pipeline vs LangChain / LangGraph

**主幹是 `core/pipeline.NativeRAG`**：檢索、prompt、Ollama 一條路看得見。

`integrations/langgraph.py` 把流程拆成 `prepare → retrieve → generate` 節點，方便對照「圖編排」；實際檢索與生成仍走 native + `ask_orchestrator`。  
`integrations/langchain.py` 僅薄包裝，沒有另建一套 Qdrant retriever。

原因：學習目標是 **RAG 零件**（chunk、embed、hybrid、rerank），不是 Agent 框架本身。

---

## Chunker

| 模式 | 適合 |
|------|------|
| `semantic`（預設） | 短句、金句密集（如納瓦尔） |
| `native` | 長篇章節敘事 |

語意切用與檢索同款 BGE 算相鄰句相似度；納瓦尔實測 semantic `min=128` 在 11 題抽樣上多數優於純字數切。  
長書可 `CHUNKER_MODE=native` 後 `--force` 重建，用 `eval.py` 對照。

---

## PDF：MarkItDown vs MinerU

| | MarkItDown | MinerU |
|--|------------|--------|
| 定位 | 輕量轉 MD | 難 PDF / 版面 |
| 依賴 | 主環境已有 | `uv sync --group mineru` |
| 失敗 | 報錯 | fallback MarkItDown |

日常迭代用 MarkItDown；檢索持續偏低且為 PDF 時可試 MinerU 重建 + `eval.py --golden`。

---

## 檢索 fallback

`retrieve_fallback.py`：起點 `hybrid_rerank` → 加寬 top_k → `hybrid` → `vector`。  
`chat /debug` 可看每輪策略；MarkItDown PDF 低分時 `ingest_hints` 會建議 MinerU。

---

## 常見誤解

| 以為 | 實際 |
|------|------|
| 筆記也在 Qdrant | 筆記在 SQLite，關鍵字注入 prompt |
| eval 評 LLM 品質 | 只評檢索（`--golden` 看 top-k 是否含關鍵句） |
| book_id 隨機 | 來自 `sample_books` 檔名 slug |

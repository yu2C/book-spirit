# Book Spirit — 學習筆記

> 給**自己複習 RAG**用：名詞、取捨、常見 QA。  
> 操作步驟見 [README.md](../README.md)；模組結構見 [ARCHITECTURE.md](../ARCHITECTURE.md)。

---

## 2. 名詞速查

| 詞 | 一句話 |
|----|--------|
| **RAG** | 先搜書再生成，答案要掛在段落上 |
| **Chunk** | 切成可檢索的小段（本專案預設 512 字 + overlap 64） |
| **Embedding** | 文字 → 向量；本專案 BGE-small-zh，查詢加 `query:` |
| **Vector 檢索** | 比「意思像不像」，不一定要有相同字 |
| **BM25** | 比「關鍵字像不像」，專名、術語 often 更穩 |
| **RRF** | 把多份排名列表合併成分數（見 §3） |
| **Hybrid** | 向量 + BM25，再用 RRF 合併 |
| **Rerank** | 對候選段落用 cross-encoder 精排（更慢更準） |
| **top_k** | 最後餵給 LLM 的段落數 |
| **Qdrant** | 本專案用的向量資料庫（見 §4） |
| **Memory** | SQLite 裡**你的筆記**，不是書的全文 |

---

## 3. RRF 是什麼？（你寫的 RPF 多半指這個）

**RRF = Reciprocal Rank Fusion（倒數排名融合）**

情境：hybrid 檢索時，**向量**有一份排名，**BM25** 有另一份排名，怎麼合成？

對每個段落 \(d\)，在每份列表裡若排在第 \(r\) 名（從 0 起算），就加：

\[
\text{score}(d) += \frac{1}{k + r + 1}
\]

本專案 `k = 60`（`core/hybrid.py` 的 `RRF_K`）。  
**直覺：** 在任一份列表裡排越前面，加越多分；兩邊都靠前的段落總分最高。  
**好處：** 不用把 BM25 分數和 cosine 分數硬湊在同一尺度。

---

## 4. 向量資料庫差異 & 為何用 Qdrant？

| 產品 | 特點 | 本 toy 為何沒選為主軸 |
|------|------|------------------------|
| **Qdrant** | 開源、本地檔案/ Docker、payload filter、Python 友善 | ✅ 預設：學習曲線平、和 FastAPI 同棧 |
| **Chroma** | 極簡、嵌入式 | 多書 collection、進階 filter 時我還想練 Qdrant API |
| **FAISS** | Meta 出品、極快、偏函式庫 | 要自己管 metadata / 持久化 / 多 collection |
| **Milvus** | 分散式、重運維 | 個人單機 toy 過重 |
| **pgvector** | 向量和業務同一 Postgres | 本專案 Postgres 只 optional 做 query log |
| **Pinecone / Weaviate 雲** | 託管、省維運 | 學習目標含「本地、可重建、無雲帳單」 |

**選 Qdrant 的三個實際理由：**

1. **本地 `qdrant_storage/`** — 刪了重建索引即可，符合「RAG 可換、筆記不丟」  
2. **payload 過濾** — `book_id`、`chapter` 子字串 filter（見 API）  
3. **和專案規模匹配** — 幾千～幾萬 chunk 單機足夠，不必上分散式

---

## 5. 為何不直接用 LangChain / LangGraph？

### LangGraph 有流程嗎？

有。`integrations/langgraph.py` 大概是：

```
retrieve → generate
```

就兩步。對讀書 Q&A 來說，**業務流程確實很簡單**，不需要複雜 state machine。

### 那為何還要自寫 `core/pipeline`？

| 原因 | 說明 |
|------|------|
| **學習** | 想看清 retrieve / prompt / Ollama 各層輸入輸出 |
| **可控** | 納瓦尔式中文章節、裁 TOC、boilerplate、多書 collection 都在自己的 code |
| **評測** | `eval.py` 只測檢索；LC 包一層反而難拆錯誤來源 |
| **對照** | LC/LG 當「同一問題的另一條路」，不是主幹 |

**結論：** 不是 LG 不好，而是**這個 toy 的目標是搞懂 RAG 零件**，不是展示 Agent 編排。流程簡單時，自寫 pipeline 更透明。

---

## 6. 為何不用「更好的 splitter」？

常見現成方案：

- LangChain `RecursiveCharacterTextSplitter`
- LlamaIndex `SentenceSplitter`
- 語意分塊（embedding 相鄰句合併）等

**本專案仍自研 `ingest/chunker.py` 的原因：**

1. **中文書常沒有 `#` 標題** — 納瓦尔是「第一部分」「第一章」純文字行，自研 regex 才接得上 `chapter` metadata。  
2. **PDF 轉 MD 的髒東西** — 重複目錄、致謝、英文 Acknowledgments，要在分塊前裁掉。  
3. **實驗變因** — chunk_size / overlap 是學習重點，寫在自己檔案裡改起來直覺。  
4. **不是不能換** — 之後可加 `CHUNKER=langchain` 對照組，用 eval 比 precision@k。

**更好的 splitter 存在，但「更好」常指英文 Wiki / 有乾淨 Markdown 的場景**；你手上的中文 PDF 未必符合那些假設。

---

## 7. 書籍格式（MarkItDown）

| 類型 | 處理 |
|------|------|
| PDF, DOCX, PPTX, XLSX, HTML, EPUB* | `MarkItDown.convert()` → `outputs/<stem>.md` |
| `.md` / `.markdown` | 複製到 outputs + `normalize_md_text` |

\* EPUB 依 MarkItDown / 系統依賴而定；失敗時先改 PDF 或自行轉 MD。

入口：`ingest/converter.py`、`ingest/formats.py`。

---

## 8. 常見自問自答（QA）

**Q: 索引建了，為什麼還說「沒有相關信息」？**  
A: 可能 (1) 檢索到致謝/版權垃圾段 (2) top_k 太少 (3) LLM 太嚴。看 chat 的引用來源是否偏題。

**Q: 重建索引會不會丟筆記？**  
A: 不會。筆記在 `data/reading_memory.db`，與 Qdrant 分離。

**Q: vector 和 hybrid 什麼時候換？**  
A: 抽象問句、換句話說 → 先 vector；專名、術語、精確詞 → 試 hybrid。用 `eval.py --preview` 比。

**Q: 為什麼生成用 Ollama、檢索用 BGE？**  
A: 任務不同——檢索要小向量模型、生成要大 LLM；可以換但不建議同一模型包辦。

**Q: 範例書 `naval-almanac.pdf` 可以 commit 嗎？**  
A: Repo 內附是為了 clone 就能跑；若公開 GitHub，請確認你有權這樣散布該 PDF（見 `sample_books/README.md`）。

**Q: `/save` 的筆記下次怎麼進回答？跟 Qdrant 同一路嗎？**  
A: **不同路。** 書段落：Qdrant + BGE 檢索 →「提供的文本內容」。筆記：SQLite `search_relevant`（關鍵字比對，最多 5 則）→ `format_notes_for_prompt` 併入 prompt 的【讀者過往筆記】區塊。兩者在 `build_prompt` 才合流，筆記**不**做向量檢索（見 `memory/store.py`、`core/pipeline.py`）。

**Q: hybrid 沒有 BM25 檔會怎樣？**  
A: `FileNotFoundError` 時 **fallback 純向量**（`core/hybrid.py`）。檔名：`qdrant_storage/bm25_<book_id>.json`。

---

## 9. 容易搞混的細節

| 常以為 | 實際 |
|--------|------|
| 筆記也在 Qdrant | 筆記在 **SQLite**；重建向量索引不刪筆記 |
| top_k 是 RRF 分數 | **top_k** = 送進 LLM 的段落**個數**；**RRF** 只用在 hybrid **排名合併** |
| book_id 隨機產生 | 來自 `sample_books` **檔名 slug**（如 `naval-almanac`） |
| 查詢字串要加 `query:` 是 API 標記 | **BGE** 訓練慣例：問句加前綴、書中 chunk 不加；影響檢索相似度 |
| eval 評整段問答好不好 | **只評檢索**（固定題 + 相似度 + 人工看 top-k）；生成好壞另用手動看引用是否 grounded |
| chunk overlap 是算 chunk 之間相似度 | **overlap** = 相鄰分塊**重疊字數**，避免句斷在邊界 |
| overview 問題 top_k 可到上千 | 本專案 overview 約 **≥8**，依 `retrieval_top_k` 規則 |

---

## 10. 實驗日誌

| 日期 | 做了什麼 | 結果 |
|------|----------|------|
| | 例如：top_k 3→5 | |
| | 例如：RETRIEVAL_MODE=hybrid | |

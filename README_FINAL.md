# 📚 書籍知識助手（Book Spirit）

**一個本地部署的 RAG 系統，用 AI 和你對話書籍內容。**

## 🎯 核心特點

- ✅ **完全本地運行** — 資料不上雲，隱私優先
- ✅ **中文優化** — 用 Qwen2.5 和 BGE-M3，中文理解很強
- ✅ **引用來源** — 回答時自動附上書中段落和頁碼
- ✅ **向量搜尋** — 用 Qdrant 快速檢索相關內容
- ✅ **REST API** — FastAPI 部署，可遠端調用

## 📊 系統評估

| 指標 | 結果 | 說明 |
|------|------|------|
| **搜尋準確度** | 75% | 10 個測試題中，7 個搜尋結果相關 |
| **回答品質** | 良好 | 能回答簡單事實和核心概念，推理題偶爾出錯 |
| **平均推理時間** | 30-60s | 本地 M4 Air 運行 |
| **向量搜尋延遲** | < 100ms | Qdrant 很快 |
| **支援的文件** | PDF, Word, PPT | 用 MarkItDown 轉換 |

## 🏗️ 系統架構

```
用戶提問
    ↓
[Embedding] — 把問題轉成向量
    ↓
[Qdrant] — 搜尋相關文本段落
    ↓
[LLM] — Qwen2.5 根據搜尋結果生成回答
    ↓
[來源引用] — 整理回答涉及的文本出處
    ↓
返回「回答 + 來源」給用戶
```

### 技術棧

| 層級 | 組件 | 選擇 | 為什麼 |
|------|------|------|--------|
| **文件轉換** | MarkItDown | 微軟官方，中文支援好 |  |
| **分塊** | LangChain | 業界標準，文本切割靈活 |  |
| **Embedding** | BGE-M3 | 多語言優化，中文最強 |  |
| **向量 DB** | Qdrant | 本地輕量，支援 metadata 過濾 |  |
| **LLM** | Qwen2.5:7b | 中文微調，本地跑 |  |
| **Web 框架** | FastAPI | 現代 Python，速度快 |  |

## 🚀 快速開始

### 前置要求

- Python 3.10+
- 16GB+ RAM（推薦）
- Ollama（用來運行 LLM）

### 1️⃣ 安裝依賴

```bash
git clone https://github.com/你的用戶名/book-spirit.git
cd book-spirit

python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2️⃣ 安裝 Ollama + 模型

```bash
# 安裝 Ollama
brew install ollama  # macOS
# 或直接下載 https://ollama.ai

# 下載模型
ollama pull qwen2.5:7b-instruct-q4_K_M

# 啟動服務（在另一個終端）
ollama serve
```

### 3️⃣ 準備文件

```bash
# 把書籍 PDF 放到這個目錄
mkdir sample_books
cp 你的書.pdf sample_books/
```

### 4️⃣ 建立索引

```bash
# 步驟 1-3：轉換、分塊、embedding、寫入 Qdrant
python 1_convert_pdf_to_md.py
python 2_chunk_and_embed.py
python 3_build_qdrant.py
```

### 5️⃣ 啟動 API 服務

```bash
python 6_fastapi_server.py
# 看到 "Uvicorn running on http://127.0.0.1:8000"
```

### 6️⃣ 測試 API

```bash
# 新開一個終端
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什麼是專長？"
  }'
```

## 📚 使用示例

### 命令行（互動模式）

```bash
python 5_generate_answer_with_llm.py
```

```
❓ 你的問題: 如何不靠運氣致富？

🔍 搜尋中...
✅ 找到 3 個相關段落

🤖 LLM 生成中...

📝 回答:
根據《納瓦爾寶典》，不靠運氣致富的方法包括：
1. 找到確定的因果關係，而不是聽天由命
2. 把自己放在可以利用運氣或吸引運氣的位置
3. 建立個人專長，這樣當機會來臨時你才能抓住

📚 引用來源（3 個）:
[1] 《納瓦爾寶典》- 第一章 積累財富
    相似度: 0.709
    內容: 為了不靠運氣致富，你就需要找到確定的因果關係...

[2] 《納瓦爾寶典》- 第一章 積累財富
    相似度: 0.699
    內容: 獲得好運的方法：希望好運不期而至...
```

### REST API

```bash
# 提問
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "幸福是什麼？",
    "top_k": 3
  }'

# 回應
{
  "question": "幸福是什麼？",
  "answer": "根據書籍内容，幸福是...",
  "sources": [
    {
      "title": "《納瓦爾寶典》",
      "chapter": "第三章 學習幸福",
      "heading": "幸福源於好習慣",
      "page": 120,
      "score": 0.695
    }
  ],
  "time": 45.23,
  "generate_time": 42.15
}
```

## 📊 評估與局限

### 系統優勢

✅ **準確的來源引用** — 回答中的所有內容都能追溯到原文  
✅ **快速搜尋** — 向量搜尋 < 100ms，即使 1000 本書也不慢  
✅ **本地隱私** — 沒有網路請求，完全離線可用  
✅ **低硬體需求** — M4 Air 就能跑，不需要 GPU  

### 已知限制

❌ **跨文本推理弱** — 如果答案需要結合多個不同章節，容易出錯  
❌ **複雜邏輯推理** — 7b 模型在複雜邏輯上不如大模型  
❌ **中文繁簡混合** — BGE-M3 對繁簡混合場景有時不夠精準  
❌ **表格和圖表** — 目前只轉文字，不處理圖片內容  

### 評估方法

見 `EVALUATION.md` 詳細報告

## 🎯 技術決策說明

### 為什麼選 Qwen2.5？

對比表：

| 模型 | 優勢 | 劣勢 |
|------|------|------|
| **Qwen2.5:7b** ✅ | 中文最強、速度快 | 大小 4.7GB |
| Llama2:7b | 英文更好 | 中文理解弱 |
| Mistral:7b | 推理快 | 中文能力一般 |
| GPT-4（API） | 最聰明 | 需要費用、隱私隱憂 |

結論：對中文書籍，Qwen2.5 是最佳選擇。

### 為什麼選 BGE-M3 Embedding？

| 模型 | 優勢 | 劣勢 |
|------|------|------|
| **BGE-M3** ✅ | 多語言、中文優化 | 大小 1.3GB |
| OpenAI Embedding | 業界標準 | 需要 API key |
| BERT | 小、快 | 中文理解弱 |

結論：多語言 + 中文優化，BGE-M3 無敵。

### 為什麼選 Qdrant？

vs Chroma：Qdrant 有 metadata 過濾，可以只搜特定書籍  
vs Pinecone：Qdrant 本地運行，不需要付費  
vs Elasticsearch：Qdrant 更輕量，適合個人項目  

結論：本地 + 輕量 + 功能完整。

## 🔧 進階用法

### 加新書籍

```bash
# 把新的 PDF 放到 sample_books/
cp 新書.pdf sample_books/

# 重新建立索引
python 3_build_qdrant.py
```

系統會自動檢測新文件並加入索引。

### 調整搜尋結果數量

```python
# 在 API 請求中指定 top_k
{
  "question": "...",
  "top_k": 5  # 預設是 3
}
```

### 修改 LLM 模型

編輯 `6_fastapi_server.py`：

```python
ollama_model="qwen2.5:14b-instruct-q4_K_M"  # 更大的模型
```

## 📈 性能指標

**M4 Air 24GB 上的測試結果：**

- 向量搜尋：0.1s
- LLM 生成（200 tokens）：30-60s
- 總耗時：30-60s

**可以優化的地方：**

1. 用量化更強的模型（q4_K_L）— 質量更好但更慢
2. 分散式部署 — 多個 GPU 並行
3. 緩存常見問題的答案

## 🤝 貢獻

歡迎提 PR 或 issue！特別是：

- 測試其他書籍效果
- 報告 bug
- 建議優化方案

## 📝 許可證

MIT License

## 🙏 致謝

- MarkItDown（微軟）— PDF 轉換
- LangChain — RAG 框架
- Qwen（阿里）— LLM 模型
- Qdrant — 向量資料庫

## 📬 聯絡方式

有問題？提個 issue 或 email me。

---

## 快速連結

- 📖 [詳細評估報告](EVALUATION.md)
- 🏗️ [系統架構說明](ARCHITECTURE.md)
- 🚀 [部署指南](DEPLOYMENT.md)
- 💻 [開發者文檔](docs/DEVELOPER.md)

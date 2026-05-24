# 📚 書籍 RAG 系統 - 完整工作流

## 快速開始（5 分鐘）

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 將 PDF 放入 sample_books 目錄
# 例如: sample_books/三體.pdf

# 3. 運行 4 個步驟
python 1_convert_pdf_to_md.py      # PDF → Markdown
python 2_chunk_and_embed.py        # 分塊測試
python 3_build_qdrant.py           # 向量化 + 寫入 DB
python 4_test_search.py            # 測試搜尋品質
```

---

## 詳細步驟說明

### 步驟 1️⃣：PDF → Markdown

```bash
python 1_convert_pdf_to_md.py 三體.pdf
```

**做什麼：**
- 用 MarkItDown（微軟官方）轉換 PDF
- 保留章節和標題結構
- 輸出到 `outputs/三體.md`

**預期結果：**
```
✅ 轉換成功！
   - 行數: 2540
   - 字數: 145000
💾 已保存到: outputs/三體.md
```

**為什麼用 MarkItDown？**
- ✅ 中文支援好
- ✅ 微軟持續維護
- ✅ 速度快（100 頁 PDF = 1–2 分鐘）
- ✅ 多格式支援（Word、PPT 也行）

---

### 步驟 2️⃣：分塊 + 結構提取

```bash
python 2_chunk_and_embed.py
```

**做什麼：**
- 測試 chunk size = 256, 512, 1024
- 比較分塊效果
- 提取章節和小節資訊

**預期結果：**
```
📊 Chunk Size = 512
   - 分塊數: 284
   - 平均長度: 510
   - 前 3 個分塊預覽:
     [0] [第一部 地球往事] Trisolaris is a fictional exoplanet...
```

**重要發現：**
- Chunk Size = 256：太碎片化，上下文丟失
- Chunk Size = 512：**最推薦**（平衡點）
- Chunk Size = 1024：太大，混雜多個概念

---

### 步驟 3️⃣：Embedding + Qdrant

```bash
python 3_build_qdrant.py
```

**做什麼：**
1. 用 BGE-Small-ZH 做 Embedding（中文優化）
2. 寫入本地 Qdrant 向量資料庫
3. 快速測試搜尋

**預期結果：**
```
✅ Embedding 完成
   耗時: 45.2 秒
   速度: 6.3 chunks/sec

✅ 集合建立: books

✅ 上傳完成: 284 個向量
```

**時間估計：**
- 100 頁 PDF（~5,000 字）：20–30 秒
- 300 頁 PDF：1–2 分鐘

---

### 步驟 4️⃣：測試搜尋品質

```bash
python 4_test_search.py
```

**做什麼：**
- 用 10 個測試問題驗證搜尋品質
- 不加 LLM，純粹看向量搜尋準確度
- 計算 precision / recall

**預期結果：**
```
❓ 問題: 三體人怎樣入侵地球？
  [1] (相似度: 0.823)
       章節: 第一部 地球往事 → 三體人警告
       內容: 三體人在 1969 年向地球發送...

  [2] (相似度: 0.756)
       章節: 第二部 黑暗森林 → 人類計畫
       內容: 地球人反擊的計畫開始...
```

---

## 硬體要求

| 配件 | 需求 | 實際測試 |
|------|------|---------|
| CPU | 任何 | M4 Air、Intel i7 都可 |
| 記憶體 | 8GB 起 | 實測 4GB 可勉強執行 |
| 磁碟 | 20GB+ | 模型 + Qdrant 持久化 ~15GB |

**M4 Air 性能：**
- 轉換 PDF：1–2 分鐘（快）
- Embedding：30–60 秒（快，有 GPU 加速）
- 搜尋：<100ms（非常快）

---

## 檔案結構

```
.
├── 1_convert_pdf_to_md.py       # PDF 轉換
├── 2_chunk_and_embed.py         # 分塊測試
├── 3_build_qdrant.py            # 向量化
├── 4_test_search.py             # 搜尋測試
├── sample_books/                # 放 PDF 在這
│   └── 三體.pdf
├── outputs/                     # 轉換後的 Markdown
│   ├── 三體.md
│   └── 三體_chunks_info.txt
└── requirements.txt
```

---

## 常見問題

### Q: Embedding 下載模型失敗？
**A:** 第一次執行會自動下載 BGE-Small-ZH（~200MB）
- 確保網路正常
- 或手動下載：`python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"`

### Q: Qdrant 持久化存儲在哪？
**A:** 預設在記憶體，如果要持久化改這一行：
```python
client = QdrantClient(path="./qdrant_storage")  # 改成檔案路徑
```

### Q: 搜尋結果不好？
**A:** 通常是分塊大小問題。試試：
1. 調整 `chunk_size` 到 768 或 1024
2. 重新運行 `2_chunk_and_embed.py` 對比

### Q: 加 LLM 推理（下一步）？
**A:** 用 Ollama 執行本地 Qwen2.5：
```bash
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

---

## 下一步（集成 LLM）

當 RAG 基礎確認無誤後，下一步是加 LLM 推理：

```python
# 簡單範例
from ollama import Client

client = Client(host='http://localhost:11434')

# 1. 搜尋向量 DB
search_results = qdrant_client.search(...)

# 2. 構造 prompt
context = "\n".join([r.payload['text'] for r in search_results])
prompt = f"基於以下內容回答問題:\n{context}\n\n問題: {user_question}"

# 3. 呼叫 LLM
response = client.generate(model="qwen2.5:7b", prompt=prompt)
```

---

## 評估指標（為什麼要測試搜尋？）

即使 LLM 再強大，如果搜尋品質差，最終回答也會爛。

**檢查清單：**
- ✅ 搜尋是否找到相關文本？
- ✅ 相似度分數是否合理（>0.7 為好）？
- ✅ 前 3 個結果是否都相關？
- ✅ 分塊大小是否影響結果？

---

## 技術決策說明

### 為什麼選 MarkItDown？
- ✅ 微軟官方，持續更新
- ✅ 中文支援好
- ✅ 多格式支援（PDF、Word、PPT）
- ❌ 表格識別不如 Docling
- 替代方案：Docling（更強但更複雜）

### 為什麼選 BGE-Small-ZH？
- ✅ 中文優化（vs 多語言）
- ✅ 快速（vs BGE-M3）
- ✅ M4 Air 上友善（低 VRAM）
- ❌ 向量質量略低於 BGE-M3
- 替代方案：MinerU（中文更強但更複雜）

### 為什麼選 Qdrant？
- ✅ 輕量級（本地可跑）
- ✅ 支援 metadata 過濾
- ✅ 查詢快（<100ms）
- ❌ 不適合超大規模（>1000 萬向量）
- 替代方案：Elasticsearch（生產環境用）

---

## 下一個挑戰

1. **評估框架** - 怎樣自動檢測回答品質？
2. **多輪對話** - 記住之前的問題上下文
3. **微調** - 根據使用者偏好自動適應
4. **API 部署** - FastAPI + Chainlit 前端

---

## 聯絡 & 反饋

遇到問題？
- 檢查日誌輸出
- 確認依賴版本
- 用簡單問題測試搜尋品質

祝你好運！🚀

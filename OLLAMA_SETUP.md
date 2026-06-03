# 🚀 Ollama 設定指南（M4 Air）

## 第 1 步：安裝 Ollama

```bash
# 方法 1：用 Homebrew（推薦）
brew install ollama

# 方法 2：直接下載
# 訪問 https://ollama.ai，下載 macOS 版本
```

驗證安裝：
```bash
ollama --version
# 應該顯示版本號
```

---

## 第 2 步：拉取模型

```bash
# 拉取 Qwen2.5:7b-q4（量化版，8GB，M4 Air 推薦）
ollama pull qwen2.5:7b-q4

# 如果速度慢，可以試試
ollama pull qwen2.5:7b  # 原始精度版（14GB）

# 檢查已下載的模型
ollama list
```

**預期下載時間：**
- 7b-q4（8GB）：3-5 分鐘（看網速）
- 7b（14GB）：5-10 分鐘

---

## 第 3 步：啟動 Ollama 服務

**新開一個終端，執行：**

```bash
ollama serve
```

應該看到：
```
2025/05/22 10:30:45 "Listening on 127.0.0.1:11434"
```

**保持這個終端運行！** 不要關閉。

---

## 第 4 步：測試 Ollama

**在另一個終端測試：**

```bash
# 簡單測試
curl http://localhost:11434/api/tags

# 應該返回已安裝的模型列表
```

或者直接測試生成：
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-q4",
  "prompt": "你好",
  "stream": false
}'
```

---

## 第 5 步：運行 RAG 系統

**回到之前的終端（code 所在的目錄）：**

```bash
# 交互問答（答完可用 /save 存筆記）
uv run python scripts/chat.py
```

---

## 常見問題

### Q1：「無法連接到 Ollama」

**原因：** `ollama serve` 沒有運行

**解決：**
```bash
# 確保另一個終端還在運行 ollama serve
# 或重新執行
ollama serve
```

### Q2：模型下載太慢

**原因：** 網路或官方伺服器慢

**解決：**
- 等待，或
- 用梯子加速，或
- 試試 `ollama pull qwen2.5:7b-q4` 的不同時間

### Q3：記憶體不足

M4 Air 24GB 跑 7b-q4 **絕對沒問題**。

如果出現：
```
out of memory
```

**檢查：**
```bash
# 看看 Ollama 占用多少記憶體
top
# 或
Activity Monitor → Memory
```

### Q4：推理很慢

Qwen2.5:7b-q4 在 M4 Air 上應該 **30-60 秒/次回答**（含搜尋時間）

如果 > 2 分鐘，檢查：
- 是否有其他程式占用 GPU
- Ollama 是否有警告信息

### Q5：想用更小的模型（更快）

```bash
# Qwen2.5:3b-q4 更小更快（但質量下降）
ollama pull qwen2.5:3b-q4

# 或設定環境變數 OLLAMA_MODEL=qwen2.5:3b-q4
```

---

## 性能預期

**M4 Air 24GB + Qwen2.5:7b-q4：**

| 動作 | 耗時 |
|------|------|
| 向量搜尋 | 0.1s |
| LLM 生成（200 tokens） | 30-60s |
| **總耗時** | **30-60s** |

這是 **完全正常的**。本地運行就是這個速度。

---

## 監控 Ollama

如果想看詳細的日誌：

```bash
# 運行時加 debug
OLLAMA_DEBUG=1 ollama serve
```

---

## 下一步

1. ✅ 確保 `ollama serve` 在運行
2. ✅ 先 `uv run python scripts/build_index.py` 建索引
3. ✅ 執行 `uv run python scripts/chat.py` 自由提問

祝你好運！🚀

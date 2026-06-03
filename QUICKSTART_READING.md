# 個人使用 — 理解・問答・記憶

> 主線：**建索引 → 問答 → `/save` 記憶**。架構說明見 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 第一次

```bash
brew install uv ollama
uv sync
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve   # 另開終端

# PDF 放入 sample_books/
uv run python scripts/build_index.py --all    # 索引所有 PDF（增量，不覆蓋其他書）
# 單本：--book <book_id>   書目：--list   封存(僅筆記)：--archive <book_id>
# 改過 PDF 後：--book <id> --force
```

---

## 日常

```bash
uv run python scripts/chat.py
```

請在**真正的終端**執行（Cursor / iTerm / Terminal），才能用 ←→ 改字、↑↓ 叫出先前問題；歷史存在 `~/.book_spirit_history`。

| 動作 | 指令 |
|------|------|
| 問書 | 直接輸入問題 |
| 說明 | `/help` 或 `/h` |
| 存筆記 | `/save` 或 `/s`（需先有一則 AI 回答） |
| 只存你的話 | `/save 我的一句心得` |
| 列書目 | `/books`（book_id 由 PDF 檔名自動 slug） |
| 只問當前書 | `/book <book_id>`（預設） |
| 跨書問原文 | `/book all` |
| 封存舊書 | `build_index.py --archive <id>`（向量刪除，筆記保留） |
| 查看偏好 | `/profile` |
| 設定偏好 | `/profile 我喜歡簡短回答` |
| 離開 | `quit` 或 `q` |

`chat.py` 啟動時也會印同一份說明。指令須**完整拼寫**（例如 `/profile`，不是 `/pofile`）。

筆記：`data/reading_memory.db`（重建 Qdrant **不會**刪）

---

## API（可選）

```bash
uv run python -m api
# http://127.0.0.1:8000/docs
```

# Chat 使用說明

安裝與建索引 → [README.md](README.md)

```bash
uv sync --group langchain
ollama serve
uv run python scripts/build_index.py --book naval-almanac
uv run python scripts/chat.py
```

在系統終端執行可沿用 `~/.book_spirit_history` 的上下鍵歷史。

## 指令

| 指令 | 說明 |
|------|------|
| （直接輸入） | 問當前書 |
| `/books` | 已索引書目（含編號） |
| `/book N` / `/book <id>` | 限定書籍 |
| `/book all` | 跨書檢索 |
| `/save` | 存上一則回答與問題 |
| `/save 心得` | 只存你的文字 |
| `/notes` / `/notes all` | 列筆記 |
| `/profile` | 讀者偏好（會注入 prompt） |
| `/debug` | 檢索 fallback 細節（或 `.env` `SHOW_RETRIEVAL_DEBUG=true`） |
| `quit` | 離開 |

封存書（只留筆記、不檢索原文）：`build_index.py --archive <book_id>`

筆記在 `data/reading_memory.db`，重建 Qdrant 不會刪。

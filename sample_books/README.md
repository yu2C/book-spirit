# 範例書籍

| 檔案 | 說明 |
|------|------|
| **`naval-almanac.pdf`** | 《納瓦爾寶典》範例（repo 內附，clone 後可直接建索引） |

```bash
uv run python scripts/build_index.py --book naval-almanac --force
uv run python scripts/chat.py
```

`book_id` 固定為 **`naval-almanac`**（來自檔名，非中文長檔名）。

## 自己的書

可再放 PDF、EPUB、DOCX、Markdown 等（見 [README 支援格式](../README.md#1-支援哪些格式)）。  
**請勿** commit 其他 PDF（`.gitignore` 已擋，僅放行 `naval-almanac.pdf`）。

## 版權

公開散布 repo 前，請確認你有權附帶並分享範例 PDF；僅供個人學習時亦請使用合法取得之檔案。

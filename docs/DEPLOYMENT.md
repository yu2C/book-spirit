# 部署與工程化（可選）

本專案**核心價值在 RAG + Memory**，Docker / CI / Postgres 用於展示「可部署、可觀測」，不是閱讀主線。

## CI

[![CI](https://github.com/yu2C/book-spirit/actions/workflows/ci.yml/badge.svg)](https://github.com/yu2C/book-spirit/actions/workflows/ci.yml)

```bash
uv sync --only-group test --no-default-groups
RAG_SKIP_INIT=1 uv run pytest -q
uv run ruff check core ingest memory eval api integrations scripts tests/
```

## Docker Compose

映像建置使用 `uv sync --frozen --group langchain`（見 `Dockerfile`），不再維護 `requirements*.txt`。

```bash
cp .env.example .env
QDRANT_URL=http://localhost:6333 uv run python scripts/build_index.py

docker compose up -d
curl http://localhost:8000/health
```

服務：`qdrant`（6333）+ `postgres`（5432）+ `api`（8000）。

設定 `DATABASE_URL` 後，`/search` 與 `/ask` 會非同步寫入 `query_logs`（與 SQLite 讀書筆記無關）。

## 環境變數

見根目錄 `.env.example`：`NOTES_DB_PATH`、`RAG_BACKEND`、`RETRIEVAL_STRATEGY`、`DATABASE_URL` 等。

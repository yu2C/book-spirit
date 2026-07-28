"""PostgreSQL query logging for FastAPI endpoints."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryLogger:
    """Optional Postgres logger; no-op when DATABASE_URL is unset."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.enabled = bool(self.database_url)
        if self.enabled:
            self._ensure_table()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self.database_url)

    def _ensure_table(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_logs (
                            id SERIAL PRIMARY KEY,
                            endpoint VARCHAR(32) NOT NULL,
                            question TEXT NOT NULL,
                            backend VARCHAR(32),
                            top_k INT,
                            result_ids JSONB,
                            latency_ms DOUBLE PRECISION,
                            retrieval_mode VARCHAR(32),
                            use_rerank BOOLEAN,
                            filters JSONB,
                            stage_timings JSONB,
                            token_usage JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    for column, col_type in (
                        ("retrieval_mode", "VARCHAR(32)"),
                        ("use_rerank", "BOOLEAN"),
                        ("filters", "JSONB"),
                        ("stage_timings", "JSONB"),
                        ("token_usage", "JSONB"),
                    ):
                        cur.execute(
                            f"ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS {column} {col_type}"
                        )
                conn.commit()
        except Exception as exc:
            logger.warning("Query log table init failed: %s", exc)
            self.enabled = False

    def check_health(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def log_query(
        self,
        *,
        endpoint: str,
        question: str,
        backend: Optional[str],
        top_k: int,
        result_ids: List[Any],
        latency_ms: float,
        retrieval_mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None,
        stage_timings: Optional[Dict[str, float]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO query_logs (
                            endpoint, question, backend, top_k, result_ids, latency_ms,
                            retrieval_mode, use_rerank, filters, stage_timings, token_usage
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            endpoint,
                            question,
                            backend,
                            top_k,
                            json.dumps(result_ids),
                            latency_ms,
                            retrieval_mode,
                            use_rerank,
                            json.dumps(filters) if filters is not None else None,
                            json.dumps(stage_timings) if stage_timings is not None else None,
                            json.dumps(token_usage) if token_usage is not None else None,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to write query log: %s", exc)

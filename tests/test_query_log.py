"""Query logger unit tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def query_log_module():
    from api import query_log as module

    return module


def test_disabled_without_database_url(query_log_module, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    logger = query_log_module.QueryLogger(database_url=None)
    assert logger.enabled is False
    logger.log_query(
        endpoint="search",
        question="test",
        backend="native",
        top_k=3,
        result_ids=[1],
        latency_ms=10.0,
    )


@patch("psycopg2.connect")
def test_log_query_when_enabled(mock_connect, query_log_module):
    conn = MagicMock()
    cur = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    mock_connect.return_value = conn

    logger = query_log_module.QueryLogger(database_url="postgresql://u:p@localhost:5432/db")
    assert logger.enabled is True
    logger.log_query(
        endpoint="ask",
        question="什麼是專長？",
        backend="native",
        top_k=3,
        result_ids=[12, 15],
        latency_ms=123.4,
        retrieval_mode="hybrid",
        use_rerank=True,
        filters={"chapter": "第一部分", "heading": "專長", "book_title": None},
        stage_timings={"retrieve_seconds": 0.12, "llm_seconds": 0.45},
        token_usage={"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
    )
    insert_sql = cur.execute.call_args_list[-1][0][0]
    assert "retrieval_mode" in insert_sql
    assert "use_rerank" in insert_sql
    assert "filters" in insert_sql
    assert "stage_timings" in insert_sql
    assert "token_usage" in insert_sql

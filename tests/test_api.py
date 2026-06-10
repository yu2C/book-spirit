"""FastAPI smoke tests with mocked backends."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(app_module):
    mock_backend = MagicMock()
    mock_backend.check_ollama_health.return_value = True
    mock_backend.retrieve.return_value = [
        {
            "text": "專長無法被教授。",
            "score": 0.85,
            "chapter": "第一部分",
            "heading": "專長",
            "page": 1,
            "book_title": "納瓦爾寶典",
            "chunk_id": 12,
        }
    ]
    mock_backend.ask.return_value = {
        "question": "什麼是專長？",
        "answer": "專長是無法被教授但可以被學習的技能。",
        "sources": [
            {
                "title": "納瓦爾寶典",
                "chapter": "第一部分",
                "heading": "專長",
                "page": 1,
                "score": 0.85,
                "text_preview": "專長無法被教授。",
                "chunk_id": 12,
            }
        ],
        "time_elapsed": 0.5,
        "llm_time": 0.3,
        "backend": "native",
    }

    app_module.backends = {"native": mock_backend, "langgraph": mock_backend}
    app_module.query_logger.enabled = False
    return TestClient(app_module.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "postgres_available" in data
    assert data["ollama_available"] is True


def test_search_response_shape(client):
    response = client.post(
        "/search",
        json={"question": "什麼是專長？", "backend": "native", "top_k": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "native"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == 12
    assert data["filters"] is None
    assert data["retrieval_mode"] == "hybrid"
    assert data["use_rerank"] is True
    assert data["retrieval_strategy"] == "hybrid_rerank"


def test_search_hybrid_rerank_strategy(client, app_module):
    app_module.backends["native"].retrieve.reset_mock()
    response = client.post(
        "/search",
        json={
            "question": "什麼是專長？",
            "backend": "native",
            "retrieval_strategy": "hybrid_rerank",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_strategy"] == "hybrid_rerank"
    assert data["retrieval_mode"] == "hybrid"
    assert data["use_rerank"] is True
    _, kwargs = app_module.backends["native"].retrieve.call_args
    assert kwargs["mode"] == "hybrid"
    assert kwargs["use_rerank"] is True


def test_search_passes_metadata_filters(client, app_module):
    app_module.backends["native"].retrieve.reset_mock()
    response = client.post(
        "/search",
        json={
            "question": "什麼是專長？",
            "backend": "native",
            "chapter": "第一部分",
            "heading": "專長",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filters"] == {
        "chapter": "第一部分",
        "heading": "專長",
        "book_title": None,
        "book_id": None,
    }
    _, kwargs = app_module.backends["native"].retrieve.call_args
    assert kwargs["filters"].chapter == "第一部分"
    assert kwargs["filters"].heading == "專長"


def test_ask_response_shape(client):
    response = client.post(
        "/ask",
        json={"question": "什麼是專長？", "backend": "native"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["llm_time"] >= 0

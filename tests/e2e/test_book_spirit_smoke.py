import os
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from tests.e2e.pages.book_spirit_page import BookSpiritPage

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("RUN_SELENIUM") != "1",
        reason="set RUN_SELENIUM=1 to run browser smoke tests",
    ),
]


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def live_server(app_module):
    backend = MagicMock()
    backend.check_ollama_health.return_value = True
    backend.retrieve.return_value = [
        {
            "text": "專長無法被教授，但能透過實作與師徒傳承學習。",
            "score": 0.91,
            "chapter": "第一章　積累財富",
            "heading": "累積專長",
            "page": 13,
            "book_title": "納瓦爾寶典",
            "chunk_id": 37,
        }
    ]
    app_module.backends = {"native": backend}
    app_module.query_logger.enabled = False

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 8
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("FastAPI test server did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def browser():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1000")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    try:
        yield driver
    finally:
        driver.quit()


def test_search_flow_shows_grounded_source(browser, live_server):
    page = BookSpiritPage(browser, live_server).open().search("什麼是專長？")

    assert "deterministic regression oracle" in page.answer_text()
    source_texts = page.source_texts()
    assert len(source_texts) == 1
    assert "第一章　積累財富" in source_texts[0]
    assert "score 0.910" in source_texts[0]
    assert "專長無法被教授，但能透過實作與師徒傳承學習。" in source_texts[0]

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mock_heavy_deps():
    for name in ("sentence_transformers", "qdrant_client", "torch"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def chunk_module():
    return load_module("chunk_embed", "2_chunk_and_embed.py")


@pytest.fixture(scope="session")
def rag_module():
    _mock_heavy_deps()
    return load_module("rag_native", "rag_native.py")


@pytest.fixture(scope="session")
def app_module():
    os.environ["RAG_SKIP_INIT"] = "1"
    _mock_heavy_deps()
    return load_module("fastapi_server", "6_fastapi_server.py")

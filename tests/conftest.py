import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mock_heavy_deps():
    for name in ("sentence_transformers", "qdrant_client", "torch"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


@pytest.fixture(scope="session")
def chunk_module():
    from ingest import chunker

    return chunker


@pytest.fixture(scope="session")
def rag_module():
    _mock_heavy_deps()
    from core import pipeline

    return pipeline


@pytest.fixture(scope="session")
def app_module():
    os.environ["RAG_SKIP_INIT"] = "1"
    _mock_heavy_deps()
    import api.app as app

    return app

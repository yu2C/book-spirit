"""Tests for ingest index skip logic."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingest.indexer import assess_build_state, sources_unchanged


def test_sources_unchanged_with_recorded_mtime():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "book.pdf"
        pdf.write_bytes(b"x")
        md = Path(tmp) / "book.md"
        md.write_text("# hi", encoding="utf-8")
        mtime = pdf.stat().st_mtime
        meta = {"source_pdf_mtime": mtime}
        assert sources_unchanged(pdf, meta, md) is True


def test_sources_unchanged_legacy_via_md():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "book.pdf"
        pdf.write_bytes(b"x")
        md = Path(tmp) / "book.md"
        md.write_text("# hi", encoding="utf-8")
        meta = {}
        assert sources_unchanged(pdf, meta, md) is True


def test_assess_build_state_skip_when_index_complete():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "naval.pdf"
        pdf.write_bytes(b"pdf")

        meta = {
            "book_title": "naval",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "num_chunks": 10,
            "built_at": "2025-01-01",
            "source_pdf_mtime": pdf.stat().st_mtime,
        }

        with patch("ingest.indexer.QDRANT_PATH", root / "qdrant_storage"):
            with patch("ingest.indexer.INDEX_META_FILE", root / "qdrant_storage/index_meta.json"):
                with patch("ingest.indexer.BM25_CORPUS_FILE", root / "qdrant_storage/bm25_corpus.json"):
                    (root / "qdrant_storage").mkdir(parents=True)
                    (root / "qdrant_storage/index_meta.json").write_text(
                        json.dumps(meta), encoding="utf-8"
                    )
                    (root / "qdrant_storage/bm25_corpus.json").write_text("[]")

                    mock_client = MagicMock()
                    mock_client.collection_exists.return_value = True
                    mock_info = MagicMock()
                    mock_info.points_count = 10
                    mock_client.get_collection.return_value = mock_info

                    with patch("ingest.indexer.create_qdrant_client", return_value=mock_client):
                        with patch("ingest.indexer.EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"):
                            state = assess_build_state(pdf)
                            assert state.skip_all is True
                            assert state.run_index is False

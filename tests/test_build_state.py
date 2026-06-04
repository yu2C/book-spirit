"""Tests for ingest index skip logic."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingest.books_registry import BookEntry
from ingest.indexer import assess_build_state, bm25_path, sources_unchanged


def test_sources_unchanged_with_recorded_mtime():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "book.pdf"
        pdf.write_bytes(b"x")
        md = Path(tmp) / "book.md"
        md.write_text("# hi", encoding="utf-8")
        mtime = pdf.stat().st_mtime
        entry = BookEntry(
            book_id="book",
            book_title="book",
            pdf_path=str(pdf),
            md_path=str(md),
            source_pdf_mtime=mtime,
        )
        assert sources_unchanged(pdf, entry, md) is True


def test_sources_unchanged_legacy_via_md():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "book.pdf"
        pdf.write_bytes(b"x")
        md = Path(tmp) / "book.md"
        md.write_text("# hi", encoding="utf-8")
        entry = BookEntry(
            book_id="book",
            book_title="book",
            pdf_path=str(pdf),
            md_path=str(md),
        )
        assert sources_unchanged(pdf, entry, md) is True


def test_assess_build_state_skip_when_index_complete():
    book_id = "naval"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "naval.pdf"
        pdf.write_bytes(b"pdf")
        md = root / "outputs" / "naval.md"
        md.parent.mkdir(parents=True)
        md.write_text("# x", encoding="utf-8")

        lib_meta = {
            "version": 2,
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "books": {book_id: {"num_chunks": 10}},
        }
        registry = {
            "books": {
                book_id: {
                    "book_id": book_id,
                    "book_title": "naval",
                    "pdf_path": str(pdf),
                    "md_path": str(md),
                    "status": "indexed",
                    "source_pdf_mtime": pdf.stat().st_mtime,
                    "num_chunks": 10,
                }
            }
        }

        with patch("ingest.indexer.QDRANT_PATH", root / "qdrant_storage"):
            with patch("ingest.indexer.INDEX_META_FILE", root / "qdrant_storage/index_meta.json"):
                with patch("ingest.books_registry.REGISTRY_FILE", root / "qdrant_storage/books_registry.json"):
                    (root / "qdrant_storage").mkdir(parents=True)
                    (root / "qdrant_storage/index_meta.json").write_text(
                        json.dumps(lib_meta), encoding="utf-8"
                    )
                    (root / "qdrant_storage/books_registry.json").write_text(
                        json.dumps(registry), encoding="utf-8"
                    )
                    bm25_path(book_id).parent.mkdir(parents=True, exist_ok=True)
                    bm25_path(book_id).write_text("[]", encoding="utf-8")

                    with patch("ingest.indexer.point_count_for_book", return_value=10):
                        with patch("ingest.indexer.EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"):
                            state = assess_build_state(pdf, book_id)
                            assert state.skip_all is True
                            assert state.run_index is False

"""Read-only index catalog: meta JSON, BM25 paths, indexed book list."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import COLLECTION_NAME


def index_meta_path() -> Path:
    from core.config import QDRANT_PATH

    return QDRANT_PATH / "index_meta.json"


def bm25_path(book_id: str) -> Path:
    from core.config import QDRANT_PATH

    return QDRANT_PATH / f"bm25_{book_id}.json"


def load_library_meta_optional() -> Optional[Dict[str, Any]]:
    meta_path = index_meta_path()
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("version") == 2:
        return raw
    from ingest.books_registry import slug_from_source

    book_title = raw.get("book_title", "legacy")
    bid = slug_from_source(
        Path(raw.get("source_path") or raw.get("source_pdf", book_title + ".pdf"))
    )
    return {
        "version": 2,
        "embedding_model": raw.get("embedding_model"),
        "collection_name": raw.get("collection_name", COLLECTION_NAME),
        "books": {
            bid: {
                "book_title": book_title,
                "num_chunks": raw.get("num_chunks", 0),
                "source_pdf": raw.get("source_pdf"),
                "source_pdf_mtime": raw.get("source_pdf_mtime"),
                "source_md": raw.get("source_md"),
                "built_at": raw.get("built_at"),
            }
        },
    }


def load_index_meta() -> Dict[str, Any]:
    """Eval / tooling: read index_meta.json."""
    meta = load_library_meta_optional()
    if not meta:
        raise FileNotFoundError("找不到索引，請先執行 scripts/build_index.py")
    return meta


def list_indexed_book_ids() -> List[str]:
    from ingest.books_registry import STATUS_INDEXED, load_registry

    reg = load_registry()
    return [bid for bid, entry in reg.items() if entry.status == STATUS_INDEXED]

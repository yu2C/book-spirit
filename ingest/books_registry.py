"""Book catalog: slug from source file, indexed / archived status, paths."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import OUTPUTS_DIR, QDRANT_PATH, SAMPLE_BOOKS_DIR
from ingest.formats import iter_sample_books

REGISTRY_FILE = QDRANT_PATH / "books_registry.json"
STATUS_INDEXED = "indexed"
STATUS_ARCHIVED = "archived"
STATUS_PENDING = "pending"


@dataclass
class BookEntry:
    book_id: str
    book_title: str
    pdf_path: str  # historical field name: absolute path to source book file
    md_path: str
    status: str = STATUS_PENDING
    source_pdf_mtime: float | None = None
    num_chunks: int = 0
    indexed_at: str | None = None

    @property
    def source_path(self) -> str:
        return self.pdf_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "book_title": self.book_title,
            "pdf_path": self.pdf_path,
            "source_path": self.pdf_path,
            "md_path": self.md_path,
            "status": self.status,
            "source_pdf_mtime": self.source_pdf_mtime,
            "num_chunks": self.num_chunks,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookEntry":
        source = data.get("source_path") or data.get("pdf_path") or ""
        return cls(
            book_id=data["book_id"],
            book_title=data.get("book_title", data["book_id"]),
            pdf_path=source,
            md_path=data.get("md_path", ""),
            status=data.get("status", STATUS_PENDING),
            source_pdf_mtime=data.get("source_pdf_mtime"),
            num_chunks=int(data.get("num_chunks") or 0),
            indexed_at=data.get("indexed_at"),
        )


def slug_from_source(source_path: Path) -> str:
    """Stable short id from source filename."""
    stem = source_path.stem.strip()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", stem, flags=re.UNICODE)
    normalized = re.sub(r"-+", "-", normalized).strip("-").lower()
    if not normalized:
        normalized = "book"
    if len(normalized) > 48:
        digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
        normalized = f"{normalized[:40].rstrip('-')}-{digest}"
    return normalized


slug_from_pdf = slug_from_source  # backward-compatible alias


def md_path_for_source(source_path: Path) -> Path:
    return OUTPUTS_DIR / f"{source_path.stem}.md"


md_path_for_pdf = md_path_for_source


def load_registry() -> Dict[str, BookEntry]:
    if not REGISTRY_FILE.exists():
        return {}
    with open(REGISTRY_FILE, encoding="utf-8") as handle:
        raw = json.load(handle)
    books = raw.get("books") or raw
    if isinstance(books, list):
        return {item["book_id"]: BookEntry.from_dict(item) for item in books}
    return {bid: BookEntry.from_dict(item) for bid, item in books.items()}


def save_registry(books: Dict[str, BookEntry]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(),
        "books": {bid: entry.to_dict() for bid, entry in sorted(books.items())},
    }
    with open(REGISTRY_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def upsert_registry_entry(entry: BookEntry) -> None:
    books = load_registry()
    books[entry.book_id] = entry
    save_registry(books)


def get_book(book_id: str) -> Optional[BookEntry]:
    return load_registry().get(book_id)


def list_sample_books() -> List[tuple[str, Path]]:
    """(book_id, source_path) for each supported file in sample_books/."""
    pairs: List[tuple[str, Path]] = []
    for source in iter_sample_books(SAMPLE_BOOKS_DIR):
        pairs.append((slug_from_source(source), source.resolve()))
    return pairs


def list_pdf_books() -> List[tuple[str, Path]]:
    """Backward-compatible: all sample books (not PDF-only)."""
    return list_sample_books()


def ensure_entry_for_source(source_path: Path, *, book_id: str | None = None) -> BookEntry:
    source_path = source_path.resolve()
    bid = book_id or slug_from_source(source_path)
    md = md_path_for_source(source_path)
    existing = get_book(bid)
    title = existing.book_title if existing and existing.book_title else source_path.stem
    if existing:
        entry = BookEntry(
            book_id=bid,
            book_title=title,
            pdf_path=str(source_path),
            md_path=str(md),
            status=existing.status,
            source_pdf_mtime=existing.source_pdf_mtime,
            num_chunks=existing.num_chunks,
            indexed_at=existing.indexed_at,
        )
    else:
        entry = BookEntry(
            book_id=bid,
            book_title=source_path.stem,
            pdf_path=str(source_path),
            md_path=str(md),
            status=STATUS_PENDING,
        )
    upsert_registry_entry(entry)
    return entry


ensure_entry_for_pdf = ensure_entry_for_source


def set_book_status(book_id: str, status: str, **updates: Any) -> BookEntry:
    books = load_registry()
    if book_id not in books:
        raise KeyError(f"未知 book_id: {book_id}")
    entry = books[book_id]
    entry.status = status
    for key, value in updates.items():
        if hasattr(entry, key):
            setattr(entry, key, value)
    upsert_registry_entry(entry)
    return entry

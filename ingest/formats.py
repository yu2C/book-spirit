"""Supported book source formats under sample_books/."""

from __future__ import annotations

from pathlib import Path

# Scanned by build_index / books_registry
SAMPLE_BOOK_EXTENSIONS = (
    ".pdf",
    ".epub",
    ".md",
    ".markdown",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".pptx",
    ".xlsx",
)

# Passed to MarkItDown.convert()
MARKITDOWN_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".epub",
        ".docx",
        ".doc",
        ".html",
        ".htm",
        ".pptx",
        ".xlsx",
    }
)

# Copied (with light normalize) into outputs/, no MarkItDown
MARKDOWN_NATIVE_EXTENSIONS = frozenset({".md", ".markdown"})


def is_sample_book(path: Path) -> bool:
    return path.suffix.lower() in SAMPLE_BOOK_EXTENSIONS


def iter_sample_books(directory: Path):
    """Yield source files in stable name order (skip README etc.)."""
    if not directory.is_dir():
        return
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and is_sample_book(p) and p.name.lower() != "readme.md"
    ]
    for path in sorted(paths, key=lambda p: p.name.lower()):
        yield path

#!/usr/bin/env python3
"""Build book index: source file → Markdown → Qdrant (+ per-book BM25)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SAMPLE_BOOKS_DIR  # noqa: E402
from ingest.books_registry import (  # noqa: E402
    ensure_entry_for_source,
    list_sample_books,
    load_registry,
    md_path_for_source,
)
from ingest.converter import convert_source_to_markdown, preview_markdown  # noqa: E402
from ingest.formats import SAMPLE_BOOK_EXTENSIONS  # noqa: E402
from ingest.indexer import (  # noqa: E402
    archive_book,
    assess_build_state,
    build_index_for_book,
)


def _print_library_status() -> None:
    reg = load_registry()
    if not reg:
        print("（尚無書目，請在 sample_books/ 放入書籍後 --all）")
        return
    print("\n📚 書目（books_registry）")
    for bid, entry in sorted(reg.items()):
        print(f"  • {bid}  [{entry.status}]  {entry.book_title[:50]}...")
        if entry.num_chunks:
            print(f"      chunks: {entry.num_chunks}")


def _process_one(book_id: str, source_path: Path, *, force: bool, preview: bool) -> None:
    state = assess_build_state(source_path, book_id, force=force)
    print(f"\n{'=' * 60}\n📚 {book_id}\n{source_path.name}")
    if state.skip_all:
        print(state.message)
        return

    print(f"ℹ️  {state.message}")

    if state.run_convert:
        print("\n=== Extract: 來源 → Markdown ===")
        md_path = convert_source_to_markdown(str(source_path))
        if preview:
            preview_markdown(md_path, max_lines=15)
    else:
        print(f"\n=== Extract: 跳過（沿用 {md_path_for_source(source_path).name}）===")

    if state.run_index:
        print("\n=== Load: chunk → embed → Qdrant ===")
        build_index_for_book(book_id, source_path, force=force)
    else:
        print("\n=== Load: 跳過 ===")


def main() -> None:
    ext_help = ", ".join(SAMPLE_BOOK_EXTENSIONS)
    parser = argparse.ArgumentParser(
        description=f"建立或更新書籍向量索引（支援 {ext_help}）"
    )
    parser.add_argument("--book", metavar="BOOK_ID", help="只處理此 book_id")
    parser.add_argument("--all", action="store_true", help="處理 sample_books/ 內所有書籍")
    parser.add_argument("--archive", metavar="BOOK_ID", help="封存：移除向量與 BM25，保留筆記")
    parser.add_argument("--list", action="store_true", help="列出書目與狀態")
    parser.add_argument("--force", action="store_true", help="強制重新轉檔並重建該書索引")
    parser.add_argument("--no-preview", action="store_true", help="轉檔後不預覽 MD")
    args = parser.parse_args()

    if args.list:
        _print_library_status()
        return

    if args.archive:
        archive_book(args.archive)
        return

    books = list_sample_books()
    if not books:
        print(f"❌ 請在 {SAMPLE_BOOKS_DIR} 放入書籍（{ext_help}）")
        sys.exit(1)

    if args.all:
        targets = books
    elif args.book:
        targets = [(b, p) for b, p in books if b == args.book]
        if not targets:
            print(f"❌ 找不到 book_id={args.book}")
            sys.exit(1)
    else:
        targets = [books[0]]

    preview = not args.no_preview
    for book_id, source_path in targets:
        ensure_entry_for_source(source_path, book_id=book_id)
        _process_one(book_id, source_path, force=args.force, preview=preview)

    _print_library_status()


if __name__ == "__main__":
    main()

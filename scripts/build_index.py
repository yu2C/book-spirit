#!/usr/bin/env python3
"""Build book index: PDF → Markdown → Qdrant (+ per-book BM25)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SAMPLE_BOOKS_DIR  # noqa: E402
from ingest.books_registry import (  # noqa: E402
    ensure_entry_for_pdf,
    list_pdf_books,
    load_registry,
)
from ingest.converter import convert_pdf_to_markdown, preview_markdown  # noqa: E402
from ingest.indexer import (  # noqa: E402
    archive_book,
    assess_build_state,
    build_index_for_book,
    md_path_for_pdf,
)


def _print_library_status() -> None:
    reg = load_registry()
    if not reg:
        print("（尚無書目，請在 sample_books/ 放入 PDF 後 --all）")
        return
    print("\n📚 書目（books_registry）")
    for bid, entry in sorted(reg.items()):
        print(f"  • {bid}  [{entry.status}]  {entry.book_title[:50]}...")
        if entry.num_chunks:
            print(f"      chunks: {entry.num_chunks}")


def _process_one(book_id: str, pdf_path: Path, *, force: bool, preview: bool) -> None:
    state = assess_build_state(pdf_path, book_id, force=force)
    print(f"\n{'=' * 60}\n📚 {book_id}\n{pdf_path.name}")
    if state.skip_all:
        print(state.message)
        return

    print(f"ℹ️  {state.message}")

    if state.run_convert:
        print("\n=== Extract: PDF → Markdown ===")
        md_path = convert_pdf_to_markdown(str(pdf_path))
        if preview:
            preview_markdown(md_path, max_lines=15)
    else:
        print(f"\n=== Extract: 跳過（沿用 {md_path_for_pdf(pdf_path).name}）===")

    if state.run_index:
        print("\n=== Load: chunk → embed → Qdrant ===")
        build_index_for_book(book_id, pdf_path, force=force)
    else:
        print("\n=== Load: 跳過 ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="建立或更新書籍向量索引（多書）")
    parser.add_argument("--book", metavar="BOOK_ID", help="只處理此 book_id（PDF slug 或 registry id）")
    parser.add_argument("--all", action="store_true", help="處理 sample_books/ 內所有 PDF")
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
        _print_library_status()
        return

    pdfs = list_pdf_books()
    if not pdfs:
        print(f"❌ 請在 {SAMPLE_BOOKS_DIR} 放入 PDF")
        sys.exit(1)

    if args.all:
        targets = pdfs
    elif args.book:
        targets = [(b, p) for b, p in pdfs if b == args.book]
        if not targets:
            print(f"❌ 找不到 book_id={args.book}（可用 --list 查看 slug）")
            sys.exit(1)
    else:
        targets = [pdfs[0]]
        print(f"ℹ️  未指定 --book / --all，僅處理第一本: {targets[0][0]}")

    preview = not args.no_preview
    for book_id, pdf_path in targets:
        ensure_entry_for_pdf(pdf_path, book_id=book_id)
        _process_one(book_id, pdf_path, force=args.force, preview=preview)

    _print_library_status()
    print("\n✅ 完成。下一步：")
    print("   uv run python scripts/chat.py")
    print("   /books  查看 book_id；/book <id> 切書；/book all 跨書")


if __name__ == "__main__":
    main()

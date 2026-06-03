#!/usr/bin/env python3
"""Build book index: PDF → Markdown → Qdrant (+ BM25 corpus)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SAMPLE_BOOKS_DIR  # noqa: E402
from ingest.converter import convert_pdf_to_markdown, preview_markdown  # noqa: E402
from ingest.indexer import assess_build_state, build_index, md_path_for_pdf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="建立或更新書籍向量索引")
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新轉 PDF 並重建 Qdrant 索引",
    )
    args = parser.parse_args()

    pdf_files = sorted(SAMPLE_BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ 請在 {SAMPLE_BOOKS_DIR} 放入 PDF")
        sys.exit(1)

    pdf_path = pdf_files[0].resolve()
    print(f"📚 PDF: {pdf_path}")

    state = assess_build_state(pdf_path, force=args.force)
    if state.skip_all:
        print(state.message)
        print("\n下一步：")
        print("   uv run python scripts/chat.py")
        return

    print(f"ℹ️  {state.message}")

    if state.run_convert:
        print("\n=== Extract: PDF → Markdown ===")
        md_path = convert_pdf_to_markdown(str(pdf_path))
        preview_markdown(md_path, max_lines=30)
    else:
        md_path = md_path_for_pdf(pdf_path)
        print(f"\n=== Extract: 跳過（沿用 {md_path.name}）===")

    if state.run_index:
        print("\n=== Load: chunk → embed → Qdrant ===")
        build_index(force=args.force, pdf_path=pdf_path)
    else:
        print("\n=== Load: 跳過（索引已就緒）===")

    print("\n✅ 完成。下一步：")
    print("   uv run python scripts/chat.py")
    print("   uv run python scripts/eval.py --preview")


if __name__ == "__main__":
    main()

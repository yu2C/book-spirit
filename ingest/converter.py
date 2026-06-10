"""
Extract: book source → Markdown in outputs/
EXTRACT_BACKEND=mineru 時 PDF 等走 MinerU CLI；否則 MarkItDown。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from markitdown import MarkItDown

from core.config import EXTRACT_BACKEND, EXTRACT_BACKEND_MINERU
from ingest.chunker import normalize_md_text
from ingest.formats import MARKDOWN_NATIVE_EXTENSIONS, MARKITDOWN_EXTENSIONS

logger = logging.getLogger(__name__)


def _convert_markitdown(src: Path, output_path: Path) -> None:
    ext = src.suffix.lower()
    if ext in MARKDOWN_NATIVE_EXTENSIONS:
        text = normalize_md_text(src.read_text(encoding="utf-8"))
        output_path.write_text(text, encoding="utf-8")
        return
    if ext not in MARKITDOWN_EXTENSIONS:
        raise ValueError(
            f"不支援的副檔名 {ext}；支援: {', '.join(sorted(MARKITDOWN_EXTENSIONS | MARKDOWN_NATIVE_EXTENSIONS))}"
        )
    converter = MarkItDown()
    result = converter.convert(str(src))
    text = normalize_md_text(result.text_content)
    output_path.write_text(text, encoding="utf-8")
    if output_path.stat().st_size == 0:
        raise RuntimeError("MarkItDown 轉換結果為空")


def convert_source_to_markdown(source_path: str, output_dir: str = "outputs") -> str:
    """
    Convert or stage a book source file to outputs/<stem>.md.
    """
    src = Path(source_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(source_path)

    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"{src.stem}.md"
    ext = src.suffix.lower()

    print(f"🔄 開始處理: {src}（extract={EXTRACT_BACKEND}）")
    print(f"📝 輸出位置: {output_path}")

    try:
        use_mineru = EXTRACT_BACKEND == EXTRACT_BACKEND_MINERU
        if use_mineru and ext not in MARKDOWN_NATIVE_EXTENSIONS:
            from ingest.mineru_extract import MINERU_SOURCE_EXTENSIONS, convert_with_mineru

            if ext in MINERU_SOURCE_EXTENSIONS:
                try:
                    return convert_with_mineru(src, output_path)
                except Exception as exc:
                    logger.warning("MinerU 失敗，改 MarkItDown: %s", exc)
                    print(f"⚠️  MinerU 失敗，改用 MarkItDown: {exc}")

        _convert_markitdown(src, output_path)
        chars = len(output_path.read_text(encoding="utf-8"))
        print("✅ 處理成功！")
        print(f"   - 字數: {chars}")
        print(f"💾 已保存到: {output_path}")
        return str(output_path)

    except FileNotFoundError:
        print(f"❌ 找不到檔案: {source_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ 轉換失敗: {exc}")
        sys.exit(1)


def convert_pdf_to_markdown(pdf_path: str, output_dir: str = "outputs") -> str:
    """Backward-compatible alias."""
    return convert_source_to_markdown(pdf_path, output_dir=output_dir)


def preview_markdown(md_path: str, max_lines: int = 50) -> None:
    with open(md_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    print(f"\n📖 預覽前 {max_lines} 行:")
    print("=" * 80)
    for line in lines[:max_lines]:
        print(line.rstrip())
    print("=" * 80)


if __name__ == "__main__":
    from core.config import SAMPLE_BOOKS_DIR
    from ingest.formats import iter_sample_books

    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        books = list(iter_sample_books(SAMPLE_BOOKS_DIR))
        if not books:
            print("❌ 使用方法:")
            print("   uv run python -m ingest.converter <書籍路徑>")
            print(f"\n   或在 {SAMPLE_BOOKS_DIR} 放支援格式的檔案")
            sys.exit(1)
        source_file = str(books[0])
        print(f"📚 找到: {source_file}")

    md_path = convert_source_to_markdown(source_file)
    preview_markdown(md_path)

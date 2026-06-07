from pathlib import Path

from ingest.converter import convert_source_to_markdown
from ingest.formats import iter_sample_books


def test_convert_native_markdown(tmp_path):
    src = tmp_path / "note.md"
    src.write_text("# 標題\n\n第一段。", encoding="utf-8")
    out_dir = tmp_path / "outputs"
    md_path = convert_source_to_markdown(str(src), output_dir=str(out_dir))
    text = Path(md_path).read_text(encoding="utf-8")
    assert "第一段" in text


def test_iter_sample_books_skips_readme(tmp_path, monkeypatch):
    from ingest import books_registry as reg

    monkeypatch.setattr(reg, "SAMPLE_BOOKS_DIR", tmp_path)
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "book.pdf").write_bytes(b"%PDF-1.4")
    books = list(iter_sample_books(tmp_path))
    assert len(books) == 1
    assert books[0].suffix == ".pdf"


from ingest.mineru_extract import _pick_markdown


def test_pick_markdown_prefers_stem(tmp_path):
    (tmp_path / "other.md").write_text("x", encoding="utf-8")
    target = tmp_path / "naval" / "naval-almanac.md"
    target.parent.mkdir()
    target.write_text("content " * 50, encoding="utf-8")
    picked = _pick_markdown(tmp_path, "naval-almanac")
    assert picked == target

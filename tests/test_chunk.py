"""Chunk function smoke tests (no GPU)."""


def test_normalize_md_text_strips_form_feed(chunk_module):
    text = "hello\fworld"
    assert "\f" not in chunk_module.normalize_md_text(text)


def test_parse_line_structure_markdown_heading(chunk_module):
    chapter, heading = chunk_module.parse_line_structure("# 第一章")
    assert chapter == "第一章"
    assert heading is None


def test_chunk_markdown_produces_chunks(chunk_module):
    md = """# 第一章 測試

這是一段測試文字。""" + ("專長是累積而來。" * 40)
    chunks = chunk_module.chunk_markdown(md, chunk_size=128, overlap=16)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_chunk_carries_chapter_metadata(chunk_module):
    md = """背景

第一部分 測試章

這是正文內容。""" + ("更多內容。" * 50)
    chunks = chunk_module.chunk_markdown(md, chunk_size=64, overlap=8)
    assert any(c.chapter for c in chunks)

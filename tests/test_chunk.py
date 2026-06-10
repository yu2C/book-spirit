"""Chunk function smoke tests (no GPU)."""

import pytest


@pytest.fixture(autouse=True)
def native_chunker_unless_semantic(monkeypatch, request):
    if "semantic" in request.node.name:
        return
    monkeypatch.setenv("CHUNKER_MODE", "native")


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


def test_chunk_markdown_heading_starts_body(chunk_module):
    md = """# 第一章 測試

正文。""" + ("專長累積。" * 40)
    chunks = chunk_module.chunk_markdown(md, chunk_size=64, overlap=8)
    assert any(c.chapter for c in chunks)


def test_chunk_semantic_splits_on_low_similarity(chunk_module):
    import numpy as np

    from ingest.chunker_semantic import chunk_markdown_semantic

    class FakeModel:
        def encode(self, texts, **kwargs):
            vecs = []
            for t in texts:
                v = np.zeros(4, dtype=float)
                if "主題A" in t:
                    v[0] = 1.0
                elif "主題B" in t:
                    v[1] = 1.0
                else:
                    v[2] = 1.0
                v = v / np.linalg.norm(v)
                vecs.append(v)
            return np.array(vecs)

    md = "主題A第一句。主題A第二句。" + ("主題A延續。" * 5) + "主題B第一句。主題B第二句。"
    chunks = chunk_markdown_semantic(
        md,
        chunk_size=512,
        overlap=0,
        threshold=0.72,
        min_chunk_size=0,
        breakpoint_percentile=50,
        model=FakeModel(),
    )
    assert len(chunks) >= 2
    assert all(c.text for c in chunks)

"""Tests for memory.store."""

import tempfile
from pathlib import Path

from memory.store import ReadingMemory, format_notes_for_prompt


def test_add_and_search_note():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        mem = ReadingMemory(db)

        mem.add_note(
            book_id="naval-almanack",
            my_take="專長是無法被教授但可以被學習的技能。",
            chapter="第一部分",
            quote="專長無法被教授",
            tags="專長,財富",
            question="什麼是專長？",
        )
        mem.set_profile("我主要關心財富篇，喜歡簡短回答。")

        hits = mem.search_relevant("什麼是專長", book_id="naval-almanack", limit=5)
        assert len(hits) >= 1
        assert "專長" in hits[0].my_take
        assert hits[0].question == "什麼是專長？"

        block = format_notes_for_prompt(hits)
        assert "我的心得" in block
        assert mem.get_profile().startswith("我主要關心")


def test_list_recent_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        mem = ReadingMemory(Path(tmp) / "t2.db")
        mem.add_note(book_id="book-a", my_take="筆記 A")
        mem.add_note(book_id="book-b", my_take="筆記 B")

        recent = mem.search_relevant("完全不相關的xyz", book_id="book-a", limit=3)
        assert len(recent) >= 1
        assert recent[0].book_id == "book-a"

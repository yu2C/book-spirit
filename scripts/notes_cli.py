#!/usr/bin/env python3
"""CLI for reading notes (SQLite memory layer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.store import ReadingMemory  # noqa: E402


def cmd_add(args):
    mem = ReadingMemory()
    note = mem.add_note(
        book_id=args.book_id,
        my_take=args.my_take,
        book_title=args.book_title or "",
        chapter=args.chapter or "",
        heading=args.heading or "",
        quote=args.quote or "",
        tags=args.tags or "",
        chunk_id=args.chunk_id,
    )
    print(f"✅ 已儲存筆記 #{note.id}（{note.book_id}）")


def cmd_list(args):
    mem = ReadingMemory()
    if args.q:
        notes = mem.search_relevant(args.q, book_id=args.book_id, limit=args.limit)
    else:
        notes = mem.list_notes(book_id=args.book_id, limit=args.limit)
    if not notes:
        print("（尚無筆記）")
        return
    for n in notes:
        print(f"\n--- #{n.id} [{n.book_id}] {n.chapter} / {n.heading}")
        if n.quote:
            print(f"原文：{n.quote[:120]}")
        print(f"心得：{n.my_take}")


def cmd_profile(args):
    mem = ReadingMemory()
    if args.set is not None:
        mem.set_profile(args.set)
        print("✅ 已更新個人簡述")
    text = mem.get_profile()
    print("【個人簡述】")
    print(text or "（尚未設定）")
    print(f"\n筆記總數：{mem.count_notes()}")


def main():
    parser = argparse.ArgumentParser(description="Book Spirit 讀書筆記")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="新增筆記")
    p_add.add_argument("--book-id", required=True, help="書籍 ID，如 naval-almanack")
    p_add.add_argument("--my-take", required=True, help="你的心得")
    p_add.add_argument("--book-title", default="")
    p_add.add_argument("--chapter", default="")
    p_add.add_argument("--heading", default="")
    p_add.add_argument("--quote", default="", help="書中原句摘錄")
    p_add.add_argument("--tags", default="")
    p_add.add_argument("--chunk-id", type=int, default=None)
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="列出或搜尋筆記")
    p_list.add_argument("--book-id", default=None)
    p_list.add_argument("-q", default=None, help="依關鍵字搜尋筆記")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_prof = sub.add_parser("profile", help="查看或設定個人簡述")
    p_prof.add_argument("--set", default=None, help="寫入偏好（例：關注財富篇、喜歡簡短回答）")
    p_prof.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

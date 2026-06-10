"""SQLite reading memory: notes and user profile survive embedding / index rebuilds."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _keywords(text: str) -> List[str]:
    """Extract Chinese runs (2+) and keep short questions as a whole."""
    text = text.strip()
    parts = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    if len(text) <= 24 and text:
        parts.append(text)
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:8]


@dataclass
class ReadingNote:
    id: int
    book_id: str
    book_title: str
    chapter: str
    heading: str
    quote: str
    my_take: str
    tags: str
    chunk_id: Optional[int]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "chapter": self.chapter,
            "heading": self.heading,
            "quote": self.quote,
            "my_take": self.my_take,
            "tags": self.tags,
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ReadingMemory:
    """Local SQLite store for reading notes and a short user profile."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or os.getenv("NOTES_DB_PATH", "data/reading_memory.db"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reading_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id TEXT NOT NULL,
                    book_title TEXT DEFAULT '',
                    chapter TEXT DEFAULT '',
                    heading TEXT DEFAULT '',
                    quote TEXT DEFAULT '',
                    my_take TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    chunk_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notes_book ON reading_notes(book_id);
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    profile_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO user_profile (id, profile_text, updated_at)
                VALUES (1, '', datetime('now'));
                """
            )
            conn.commit()

    def add_note(
        self,
        *,
        book_id: str,
        my_take: str,
        book_title: str = "",
        chapter: str = "",
        heading: str = "",
        quote: str = "",
        tags: str = "",
        chunk_id: Optional[int] = None,
    ) -> ReadingNote:
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reading_notes (
                    book_id, book_title, chapter, heading, quote, my_take, tags,
                    chunk_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    book_title,
                    chapter,
                    heading,
                    quote,
                    my_take,
                    tags,
                    chunk_id,
                    now,
                    now,
                ),
            )
            conn.commit()
            note_id = cur.lastrowid
        note = self.get_note(note_id)
        if note is None:
            raise RuntimeError("Failed to read inserted note")
        return note

    def get_note(self, note_id: int) -> Optional[ReadingNote]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reading_notes WHERE id = ?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def list_notes(
        self,
        *,
        book_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ReadingNote]:
        sql = "SELECT * FROM reading_notes"
        params: list[Any] = []
        if book_id:
            sql += " WHERE book_id = ?"
            params.append(book_id)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_note(r) for r in rows]

    def search_relevant(
        self,
        question: str,
        *,
        book_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[ReadingNote]:
        """Keyword match on notes; fallback to recent notes for the book."""
        keywords = _keywords(question)
        if not keywords:
            return self.list_notes(book_id=book_id, limit=limit)

        clauses = []
        params: list[Any] = []
        for kw in keywords:
            pattern = f"%{kw}%"
            clauses.append(
                "(my_take LIKE ? OR quote LIKE ? OR chapter LIKE ? OR heading LIKE ? OR tags LIKE ?)"
            )
            params.extend([pattern] * 5)
        where = "(" + " OR ".join(clauses) + ")"
        if book_id:
            where = "book_id = ? AND " + where
            params = [book_id, *params]

        sql = f"""
            SELECT * FROM reading_notes
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if rows:
            return [self._row_to_note(r) for r in rows]
        return self.list_notes(book_id=book_id, limit=limit)

    def get_profile(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT profile_text FROM user_profile WHERE id = 1").fetchone()
        return row["profile_text"] if row else ""

    def set_profile(self, profile_text: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_profile SET profile_text = ?, updated_at = ?
                WHERE id = 1
                """,
                (profile_text, now),
            )
            conn.commit()

    def count_notes(self, book_id: Optional[str] = None) -> int:
        with self._connect() as conn:
            if book_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM reading_notes WHERE book_id = ?",
                    (book_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM reading_notes").fetchone()
        return int(row["c"]) if row else 0

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> ReadingNote:
        return ReadingNote(
            id=row["id"],
            book_id=row["book_id"],
            book_title=row["book_title"] or "",
            chapter=row["chapter"] or "",
            heading=row["heading"] or "",
            quote=row["quote"] or "",
            my_take=row["my_take"],
            tags=row["tags"] or "",
            chunk_id=row["chunk_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def format_notes_for_prompt(notes: List[ReadingNote]) -> str:
    if not notes:
        return ""
    lines = ["【讀者過往筆記（你的理解，可輔助回答，但書中引述仍須以「提供的文本內容」為準）】"]
    for i, n in enumerate(notes, 1):
        loc = " / ".join(x for x in (n.book_title, n.chapter, n.heading) if x)
        lines.append(f"\n[筆記 {i}] {loc or n.book_id}")
        if n.quote:
            lines.append(f"原文摘錄：{n.quote}")
        lines.append(f"我的心得：{n.my_take}")
        if n.tags:
            lines.append(f"標籤：{n.tags}")
    return "\n".join(lines)

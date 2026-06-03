#!/usr/bin/env python3
"""Interactive Q&A with /save for reading notes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HISTORY_FILE = Path.home() / ".book_spirit_history"
HISTORY_MAX = 500


def _import_readline():
    try:
        import readline

        return readline
    except ImportError:
        try:
            import gnureadline as readline  # type: ignore[no-redef]

            return readline
        except ImportError:
            return None


def _setup_readline() -> bool:
    """Enable line editing (arrow keys, Home/End) in interactive terminals."""
    if not sys.stdin.isatty():
        return False
    readline = _import_readline()
    if readline is None:
        return False
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(HISTORY_MAX)
    return True


def _save_readline_history() -> None:
    readline = _import_readline()
    if readline is None or not sys.stdin.isatty():
        return
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def _prompt() -> str:
    return input("\n❓ ").strip()


from core.pipeline import NativeRAG  # noqa: E402
from memory.store import ReadingMemory  # noqa: E402

SLASH_COMMANDS = ("/help", "/h", "/save", "/s", "/book", "/profile")

SLASH_HELP = """
📌 斜線指令（先問答再 /save；隨時可用 /book、/profile、/help）：
  /help 或 /h        顯示本說明
  /save 或 /s        把上一則 AI 回答存成筆記
  /save 你的心得     心得用你寫的這句（書摘仍取自上一則來源）
  /book 書籍id       設定目前書籍（例：/book naval-almanack）
  /book              只顯示目前 book_id
  /profile           查看個人簡述
  /profile 文字      設定簡述（會影響之後回答）
  quit 或 q          離開
"""


def _print_answer(result: dict) -> None:
    print(f"\n{'=' * 80}\n📝 回答\n{'=' * 80}\n")
    print(result["answer"])
    sources = result.get("sources") or []
    if sources:
        print(f"\n{'=' * 80}\n📚 引用來源（{len(sources)} 個）\n{'=' * 80}\n")
        for i, source in enumerate(sources, 1):
            title = source.get("title") or source.get("book_title") or ""
            print(f"[{i}] {title}")
            print(f"    {source.get('chapter', '')} → {source.get('heading', '')}")
            print(f"    相似度: {source.get('score', 0):.3f}")
            preview = source.get("text_preview") or ""
            if preview:
                print(f"    預覽: {preview}...\n")
    if result.get("memory_notes_used"):
        print(f"💭 已參考 {len(result['memory_notes_used'])} 則過往筆記")
    print(
        f"\n⏱️  {result['time_elapsed']:.2f}s (LLM: {result.get('llm_time', 0):.2f}s)"
    )
    print("👉 覺得不錯可輸入 /save 或 /s")


def _save_note(last_result: dict | None, memory: ReadingMemory, book_id: str, custom: str | None) -> None:
    if not last_result or not last_result.get("answer") or str(last_result["answer"]).startswith("❌"):
        print("❌ 尚無可儲存的有效回答")
        return
    src = (last_result.get("sources") or [{}])[0]
    note = memory.add_note(
        book_id=book_id,
        my_take=(custom or last_result["answer"]).strip(),
        book_title=src.get("title") or "",
        chapter=src.get("chapter") or "",
        heading=src.get("heading") or "",
        quote=(src.get("text_preview") or "").strip(),
        chunk_id=src.get("chunk_id"),
    )
    print(f"✅ 筆記 #{note.id}（{book_id}）")


def _suggest_slash_command(cmd: str) -> str | None:
    from difflib import get_close_matches

    hit = get_close_matches(cmd, SLASH_COMMANDS, n=1, cutoff=0.75)
    return hit[0] if hit else None


def _slash(line: str, last_result: dict | None, memory: ReadingMemory, book_id: str) -> str:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    if cmd not in SLASH_COMMANDS and not cmd.startswith("/profile"):
        hint = _suggest_slash_command(cmd)
        if hint:
            print(f"❌ 未知指令 {cmd}，你是否要打 {hint}？輸入 /help 查看")
            return book_id
    if cmd in ("/help", "/h"):
        print(SLASH_HELP)
    elif cmd in ("/save", "/s"):
        custom = parts[1].strip() if len(parts) > 1 else None
        _save_note(last_result, memory, book_id, custom)
    elif cmd == "/book":
        if len(parts) > 1:
            book_id = parts[1].strip()
            print(f"✅ book_id = {book_id}")
        else:
            print(f"目前 book_id = {book_id}")
    elif cmd == "/profile":
        rest = line.strip()[len("/profile") :].strip()
        if rest:
            memory.set_profile(rest)
            print("✅ 已更新 profile")
        print(memory.get_profile() or "（尚未設定）")
    else:
        print(f"❌ 未知指令 {cmd}，/help 查看")
    return book_id


def main() -> None:
    rag = NativeRAG()
    memory = ReadingMemory()
    book_id = os.getenv("DEFAULT_BOOK_ID", "naval-almanack")

    if not rag.check_ollama_health():
        print("❌ 請先執行：ollama serve")
        sys.exit(1)

    print("📚 Book Spirit 問答")
    print(f"book_id: {book_id}")
    print(SLASH_HELP)
    if _setup_readline():
        print("⌨️  已啟用終端行編輯（←→ 移動游標，↑↓ 歷史輸入）")

    last: dict | None = None
    try:
        while True:
            line = _prompt()
            if line.lower() in ("quit", "exit", "q"):
                break
            if not line:
                continue
            if line.startswith("/"):
                book_id = _slash(line, last, memory, book_id)
                continue
            last = rag.ask(line, book_id=book_id, use_memory=True)
            _print_answer(last)
    finally:
        _save_readline_history()
    print("\n👋 再見")


if __name__ == "__main__":
    main()

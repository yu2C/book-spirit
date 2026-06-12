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


from core.backends import build_rag_backend  # noqa: E402
from core.library_scope import (  # noqa: E402
    format_scope_label,
    indexed_books_menu,
    resolve_book_arg,
)
from memory.store import ReadingMemory  # noqa: E402

SLASH_COMMANDS = (
    "/help",
    "/h",
    "/save",
    "/s",
    "/book",
    "/books",
    "/notes",
    "/profile",
    "/debug",
)

_show_retrieval_debug = os.getenv("SHOW_RETRIEVAL_DEBUG", "false").lower() in ("1", "true", "yes")

SLASH_HELP = """
📌 斜線指令（先問答再 /save）：
  /help 或 /h        顯示本說明
  /books             已索引書籍（編號）；未索引另列無編號
  /book N            用編號切書（見 /books）
  /book 書籍id       用 slug 切書
  /book all          跨書檢索
  /book              顯示目前範圍
  /notes             最近筆記（當前書）
  /notes all         最近筆記（全部）
  /save 或 /s        存上一則回答（含問題）
  /profile 文字      讀者偏好
  /debug             切換檢索 debug
  quit 或 q          離開
"""


def _default_book_id() -> str:
    from core.index_catalog import list_indexed_book_ids
    from ingest.books_registry import list_pdf_books

    indexed = list_indexed_book_ids()
    if indexed:
        return sorted(indexed)[0]
    pdfs = list_pdf_books()
    if pdfs:
        return pdfs[0][0]
    return os.getenv("DEFAULT_BOOK_ID", "naval-almanack")


def _print_books() -> None:
    from ingest.books_registry import STATUS_INDEXED, load_registry

    menu = indexed_books_menu()
    if not menu:
        print("（尚無已索引書籍 → uv run python scripts/build_index.py --all）")
    else:
        print("📗 已索引（可用 /book N 切換）：")
        for n, bid, title in menu:
            print(f"  [{n}] {title}")
            print(f"      {bid}")
    reg = load_registry()
    pending = [
        (bid, e)
        for bid, e in sorted(reg.items())
        if e.status != STATUS_INDEXED
    ]
    if pending:
        print("\n📦 未索引（需 build_index，無編號）：")
        for bid, entry in pending:
            print(f"  {bid}  [{entry.status}]  {entry.book_title[:60]}")


def _print_notes(memory: ReadingMemory, book_id: str) -> None:
    if book_id == "all":
        notes = memory.list_notes(limit=10)
        header = "全部書籍"
    else:
        notes = memory.list_notes(book_id=book_id, limit=10)
        header = format_scope_label(book_id)
    if not notes:
        print(f"（{header} 尚無筆記）")
        return
    print(f"📝 最近筆記 — {header}（最多 10 則）")
    for n in notes:
        q = f"  問：{n.question[:48]}…" if len(n.question) > 48 else f"  問：{n.question}"
        take = n.my_take.replace("\n", " ")
        preview = take[:72] + "…" if len(take) > 72 else take
        loc = " / ".join(x for x in (n.chapter, n.heading) if x)
        print(f"\n  #{n.id}  {loc or n.book_id}")
        if n.question:
            print(q)
        print(f"  心得：{preview}")


def _print_answer(result: dict) -> None:
    print(f"\n{'=' * 80}\n📝 回答\n{'=' * 80}\n")
    print(result["answer"])
    sources = result.get("sources") or []
    if sources:
        print(f"\n{'=' * 80}\n📚 引用來源（{len(sources)} 個）\n{'=' * 80}\n")
        for i, source in enumerate(sources, 1):
            title = source.get("title") or source.get("book_title") or ""
            print(f"[來源 {i}] {title}")
            print(f"    {source.get('chapter', '')} → {source.get('heading', '')}")
            print(f"    相似度: {source.get('score', 0):.3f}")
            preview = source.get("text_preview") or ""
            if preview:
                print(f"    預覽: {preview}...\n")
    if result.get("memory_notes_used"):
        print(f"💭 已參考 {len(result['memory_notes_used'])} 則過往筆記")
    if _show_retrieval_debug:
        queries = result.get("retrieval_queries_used") or []
        if queries:
            print(f"\n🔍 檢索用查詢: {' | '.join(queries)}")
        rdebug = result.get("retrieval_debug") or {}
        attempts = rdebug.get("attempts") or []
        if attempts:
            print("🔄 檢索 fallback:")
            for att in attempts:
                mark = "✓" if att.get("sufficient") else "·"
                print(
                    f"   [{att.get('attempt')}] {mark} {att.get('strategy')} "
                    f"top_k={att.get('top_k')} top1={att.get('top1_score', 0):.3f} "
                    f"n={att.get('num_docs')}"
                )
            if rdebug.get("final_strategy"):
                print(f"   → 最終: {rdebug['final_strategy']}")
    hint = result.get("ingest_hint")
    if hint:
        print(f"\n{hint}")
    print(f"\n⏱️  {result['time_elapsed']:.2f}s (LLM: {result.get('llm_time', 0):.2f}s)")
    print("👉 覺得不錯可輸入 /save 或 /s")


def _note_book_id(scope_book_id: str, last_result: dict) -> str:
    if scope_book_id != "all":
        return scope_book_id
    src = (last_result.get("sources") or [{}])[0]
    return src.get("book_id") or scope_book_id


def _save_note(
    last_result: dict | None, memory: ReadingMemory, book_id: str, custom: str | None
) -> None:
    if (
        not last_result
        or not last_result.get("answer")
        or str(last_result["answer"]).startswith("❌")
    ):
        print("❌ 尚無可儲存的有效回答")
        return
    if book_id == "all":
        note_book = _note_book_id(book_id, last_result)
        if note_book == "all":
            print("❌ 跨書問答但無來源，無法判定筆記所屬書籍")
            return
    else:
        note_book = book_id
    src = (last_result.get("sources") or [{}])[0]
    note = memory.add_note(
        book_id=note_book,
        my_take=(custom or last_result["answer"]).strip(),
        book_title=src.get("title") or "",
        chapter=src.get("chapter") or "",
        heading=src.get("heading") or "",
        quote=(src.get("text_preview") or "").strip(),
        chunk_id=src.get("chunk_id"),
        question=(last_result.get("question") or "").strip(),
    )
    print(f"✅ 筆記 #{note.id}（{format_scope_label(note_book)}）")


def _suggest_slash_command(cmd: str) -> str | None:
    from difflib import get_close_matches

    candidates = list(SLASH_COMMANDS) + ["/notes all"]
    hit = get_close_matches(cmd, candidates, n=1, cutoff=0.75)
    return hit[0] if hit else None


def _slash(line: str, last_result: dict | None, memory: ReadingMemory, book_id: str) -> str:
    global _show_retrieval_debug
    stripped = line.strip()
    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    if cmd == "/notes" and len(parts) > 1 and parts[1].strip().lower() == "all":
        _print_notes(memory, "all")
        return book_id
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
    elif cmd == "/books":
        _print_books()
    elif cmd == "/book":
        if len(parts) > 1:
            resolved, msg = resolve_book_arg(parts[1])
            print(msg)
            if resolved is not None:
                book_id = resolved
        else:
            print(f"目前檢索：{format_scope_label(book_id)}")
    elif cmd == "/notes":
        if book_id == "all":
            print("❌ 跨書模式請用 /notes all；或 /book N 指定單書後再 /notes")
        else:
            _print_notes(memory, book_id)
    elif cmd == "/profile":
        rest = stripped[len("/profile") :].strip()
        if rest:
            memory.set_profile(rest)
            print("✅ 已更新 profile")
        print(memory.get_profile() or "（尚未設定）")
    elif cmd == "/debug":
        _show_retrieval_debug = not _show_retrieval_debug
        state = "開啟" if _show_retrieval_debug else "關閉"
        print(f"✅ 檢索 debug: {state}（Planner 查詢 + fallback 輪次 + ingest 提示）")
    else:
        print(f"❌ 未知指令 {cmd}，/help 查看")
    return book_id


def main() -> None:
    try:
        rag = build_rag_backend()
    except ImportError as exc:
        print(f"❌ {exc}")
        print("   請執行：uv sync --group langchain")
        sys.exit(1)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    memory = ReadingMemory()
    book_id = os.getenv("DEFAULT_BOOK_ID") or _default_book_id()

    if not rag.check_ollama_health():
        print("❌ 請先執行：ollama serve")
        sys.exit(1)

    print(f"📚 Book Spirit 問答（backend: {os.getenv('RAG_BACKEND', 'langgraph')}）")
    print(f"檢索範圍: {format_scope_label(book_id)}（/books 看編號）")
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

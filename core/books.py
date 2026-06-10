"""Backward-compatible re-exports — prefer core.library_scope."""

from core.library_scope import BOOK_SCOPE_ALL, AskBookContext, resolve_ask_context

__all__ = ["BOOK_SCOPE_ALL", "AskBookContext", "resolve_ask_context"]

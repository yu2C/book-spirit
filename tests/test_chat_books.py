from core.chat_books import resolve_book_arg


def test_resolve_book_number_and_all(monkeypatch):
    monkeypatch.setattr(
        "core.chat_books.indexed_books_menu",
        lambda: [(1, "naval-almanac", "Naval"), (2, "other", "Other")],
    )
    bid, _ = resolve_book_arg("1")
    assert bid == "naval-almanac"
    bid, _ = resolve_book_arg("all")
    assert bid == "all"


def test_resolve_invalid_number(monkeypatch):
    monkeypatch.setattr("core.chat_books.indexed_books_menu", lambda: [(1, "a", "A")])
    bid, msg = resolve_book_arg("9")
    assert bid is None
    assert "無此編號" in msg

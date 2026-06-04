from core.collections import collection_name_for_book, vector_search_targets


def test_collection_name_for_book():
    name = collection_name_for_book("naval-almanack")
    assert name.startswith("book_")
    assert "naval" in name


def test_vector_search_targets_single():
    targets = vector_search_targets("my-book")
    assert len(targets) == 1
    assert targets[0][1] == "my-book"

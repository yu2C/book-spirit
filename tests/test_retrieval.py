from core.retrieval import (
    expand_retrieval_queries,
    is_boilerplate_chunk,
    is_overview_question,
    is_thematic_question,
    retrieval_top_k,
)


def test_overview_question_detection():
    assert is_overview_question("這本書的大綱是什麼")
    assert is_overview_question("main narrative of the book")
    assert not is_overview_question("公開信內容是什麼")


def test_retrieval_top_k_boost():
    assert retrieval_top_k("大綱", 3) >= 8
    assert retrieval_top_k("理想主義的概念", 3) >= 6
    assert retrieval_top_k("公開信", 3) >= 5


def test_thematic_expansion():
    qs = expand_retrieval_queries("書裡理想主義的概念")
    assert is_thematic_question("idealism versus greed")
    assert any("idealism" in q.lower() for q in qs)


def test_boilerplate_chunk():
    assert is_boilerplate_chunk("Acknowledgments\nThank you to my parents")
    assert is_boilerplate_chunk("I.F. STONE, proprietor of I. F.")
    assert not is_boilerplate_chunk(
        "After Arthur's call for the open letter, a number of people admonished him."
    )


def test_expand_queries_for_overview():
    qs = expand_retrieval_queries("跟我講本書的主要敘事")
    assert len(qs) >= 2
    assert any("introduction" in q.lower() for q in qs)

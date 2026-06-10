from core.ingest_hints import golden_ingest_warnings, ingest_hint_for_book


def test_ingest_hint_markitdown_pdf_low_retrieval():
    meta = {
        "books": {
            "naval-almanac": {
                "extract_backend": "markitdown",
                "source_path": "/books/naval-almanac.pdf",
            }
        }
    }
    hint = ingest_hint_for_book("naval-almanac", meta=meta, low_retrieval=True)
    assert hint is not None
    assert "mineru" in hint.lower()


def test_ingest_hint_skips_mineru_backend():
    meta = {
        "books": {
            "naval-almanac": {
                "extract_backend": "mineru",
                "source_path": "/books/naval-almanac.pdf",
            }
        }
    }
    assert ingest_hint_for_book("naval-almanac", meta=meta, low_retrieval=True) is None


def test_golden_warnings_when_low_pass():
    meta = {
        "books": {
            "naval-almanac": {
                "extract_backend": "markitdown",
                "source_pdf": "x.pdf",
            }
        }
    }
    warns = golden_ingest_warnings(meta, pass_rate=0.5, case_book_ids=["naval-almanac"])
    assert len(warns) == 1
    assert "mineru" in warns[0].lower()

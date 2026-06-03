"""Native RAG helpers with mocked embed/search (no GPU)."""

from unittest.mock import MagicMock, patch


def test_hits_to_docs_format(rag_module):
    point = MagicMock()
    point.id = 42
    point.score = 0.91
    point.payload = {
        "text": "測試段落",
        "chapter": "第一章",
        "heading": "專長",
        "page": 3,
        "book_title": "測試書",
        "chunk_id": 7,
    }
    docs = rag_module.hits_to_docs([point])
    assert len(docs) == 1
    assert docs[0]["score"] == 0.91
    assert docs[0]["chunk_id"] == 7
    assert docs[0]["book_title"] == "測試書"


def test_doc_to_source_preview(rag_module):
    source = rag_module.doc_to_source(
        {
            "book_title": "納瓦爾寶典",
            "chapter": "第一部分",
            "heading": "專長",
            "page": 1,
            "score": 0.88,
            "text": "x" * 200,
            "chunk_id": 3,
        }
    )
    assert source["title"] == "納瓦爾寶典"
    assert source["chunk_id"] == 3
    assert len(source["text_preview"]) == 100


def test_retrieve_uses_query_prefix(rag_module):
    with patch.object(rag_module.NativeRAG, "__init__", lambda self, **kwargs: None):
        rag = rag_module.NativeRAG()
    rag.embedding_model = MagicMock()
    vector = MagicMock()
    vector.tolist.return_value = [0.1, 0.2]
    rag.embedding_model.encode.return_value = [vector]
    rag.qdrant_client = MagicMock()
    rag.qdrant_client.query_points.return_value = MagicMock(points=[])

    rag.retrieve("如何致富？", top_k=2)

    rag.embedding_model.encode.assert_called_once()
    args, kwargs = rag.embedding_model.encode.call_args
    assert args[0] == ["query: 如何致富？"]
    assert kwargs.get("normalize_embeddings") is True


def test_retrieve_passes_query_filter(rag_module):
    from core.config import SearchFilters

    with patch.object(rag_module.NativeRAG, "__init__", lambda self, **kwargs: None):
        rag = rag_module.NativeRAG()
    rag.embedding_model = MagicMock()
    vector = MagicMock()
    vector.tolist.return_value = [0.1, 0.2]
    rag.embedding_model.encode.return_value = [vector]
    rag.qdrant_client = MagicMock()
    rag.qdrant_client.query_points.return_value = MagicMock(points=[])

    filters = SearchFilters(chapter="第一部分")
    with patch.object(rag_module, "build_qdrant_filter", return_value="MOCK_FILTER") as mock_build:
        rag.retrieve("專長", top_k=3, filters=filters)

    mock_build.assert_called_once_with(filters)
    _, kwargs = rag.qdrant_client.query_points.call_args
    assert kwargs["query_filter"] == "MOCK_FILTER"
    assert kwargs["limit"] == 30


def test_apply_payload_filters(rag_module):
    from core.config import SearchFilters

    docs = [
        {"chapter": "第一部分", "heading": "專長", "book_title": "納瓦爾寶典"},
        {"chapter": "第二部分", "heading": "財富", "book_title": "納瓦爾寶典"},
    ]
    filtered = rag_module.apply_payload_filters(
        docs, SearchFilters(chapter="第一部分", heading="專長")
    )
    assert len(filtered) == 1
    assert filtered[0]["heading"] == "專長"

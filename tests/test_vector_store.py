import pytest
from unittest.mock import patch, MagicMock
from app.rag.vector_store import init_chroma, get_collection, store_chunks, delete_document

def test_init_chroma():
    mock_client = MagicMock()
    with patch("chromadb.HttpClient", return_value=mock_client):
        init_chroma()
        mock_client.get_or_create_collection.assert_called_once()
        collection = get_collection()
        assert collection is not None

def test_store_chunks():
    mock_collection = MagicMock()
    with patch("app.rag.vector_store._collection", mock_collection):
        store_chunks(
            chunk_ids=["id1"],
            embeddings=[[0.1, 0.2]],
            documents=["text1"],
            metadatas=[{"k": "v"}]
        )
        mock_collection.add.assert_called_once_with(
            ids=["id1"],
            embeddings=[[0.1, 0.2]],
            documents=["text1"],
            metadatas=[{"k": "v"}]
        )

def test_delete_document():
    mock_collection = MagicMock()
    with patch("app.rag.vector_store._collection", mock_collection):
        delete_document("doc1", "user1")
        mock_collection.delete.assert_called_once_with(
            where={
                "$and": [
                    {"document_id": "doc1"},
                    {"user_id": "user1"}
                ]
            }
        )

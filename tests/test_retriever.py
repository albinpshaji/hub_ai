import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.rag.retriever import retrieve_chunks
from app.rag.context_assembler import assemble_context

@pytest.mark.asyncio
async def test_retrieve_chunks():
    mock_collection = MagicMock()
    # Mocking ChromaDB's return shape
    mock_collection.query.return_value = {
        "documents": [["chunk1", "chunk2", "chunk1"]]
    }
    
    with patch("app.rag.retriever.generate_embedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1, 0.2]
        
        with patch("app.rag.retriever.get_collection", return_value=mock_collection):
            chunks = await retrieve_chunks("test query", "user1")
            
            # Deduplication should reduce the 3 returned chunks to 2
            assert len(chunks) == 2
            assert chunks == ["chunk1", "chunk2"]
            
            mock_collection.query.assert_called_once_with(
                query_embeddings=[[0.1, 0.2]],
                n_results=5,
                where={"user_id": "user1"}
            )

@pytest.mark.asyncio
async def test_retrieve_chunks_empty_query():
    chunks = await retrieve_chunks("   ", "user1")
    assert chunks == []

def test_assemble_context():
    chunks = ["First chunk.", "Second chunk."]
    context = assemble_context(chunks)
    assert "--- DOCUMENT CONTEXT ---" in context
    assert "[Excerpt 1]\nFirst chunk." in context
    assert "[Excerpt 2]\nSecond chunk." in context
    assert "------------------------" in context

def test_assemble_context_empty():
    assert assemble_context([]) == ""

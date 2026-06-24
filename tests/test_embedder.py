import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.rag.embedder import generate_embedding, generate_embeddings_batch

@pytest.mark.asyncio
async def test_generate_embedding_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status.return_value = None
    
    mock_post = AsyncMock(return_value=mock_response)
    
    with patch("httpx.AsyncClient.post", new=mock_post):
        embedding = await generate_embedding("hello world")
        assert len(embedding) == 3
        assert embedding[0] == 0.1

@pytest.mark.asyncio
async def test_generate_embeddings_batch():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status.return_value = None
    
    mock_post = AsyncMock(return_value=mock_response)
    
    with patch("httpx.AsyncClient.post", new=mock_post) as mock_post_call:
        embeddings = await generate_embeddings_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert mock_post_call.call_count == 2

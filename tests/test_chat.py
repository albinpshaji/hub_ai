import pytest
from httpx import AsyncClient
from app.main import app
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_chat_stream_validation_error():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/stream",
            json={
                "messages": [],
                "user_id": "user123",
                "use_rag": True
            }
        )
        assert response.status_code == 400
        assert "Messages array cannot be empty" in response.text

@pytest.mark.asyncio
async def test_chat_stream_no_rag():
    async def mock_generator(messages):
        yield "data: {\"content\": \"hello\"}\n\n"
        yield "data: [DONE]\n\n"

    with patch("app.routers.chat.ollama_stream_generator", side_effect=mock_generator) as mock_gen:
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "user_id": "user123",
                    "use_rag": False
                }
            )
            assert response.status_code == 200
            
            # Read streaming response
            text = ""
            async for chunk in response.aiter_text():
                text += chunk
                
            assert "data: {\"content\": \"hello\"}" in text
            assert "data: [DONE]" in text
            
            # Verify RAG was not used
            mock_gen.assert_called_once()
            called_messages = mock_gen.call_args[0][0]
            assert len(called_messages) == 1
            assert called_messages[0]["content"] == "Hi"

@pytest.mark.asyncio
async def test_chat_stream_with_rag():
    async def mock_generator(messages):
        yield "data: {\"content\": \"augmented response\"}\n\n"

    with patch("app.routers.chat.ollama_stream_generator", side_effect=mock_generator) as mock_gen:
        with patch("app.routers.chat.retrieve_chunks", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = ["chunk 1 context"]
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/chat/stream",
                    json={
                        "messages": [{"role": "user", "content": "What is this?"}],
                        "user_id": "user123",
                        "use_rag": True
                    }
                )
                assert response.status_code == 200
                
                # Verify RAG was used
                mock_retrieve.assert_called_once_with("What is this?", "user123")
                
                called_messages = mock_gen.call_args[0][0]
                assert "Use the following context" in called_messages[0]["content"]
                assert "chunk 1 context" in called_messages[0]["content"]
                assert "What is this?" in called_messages[0]["content"]

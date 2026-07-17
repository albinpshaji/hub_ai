import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.search_service import search_tavily, unified_web_search
from app.config import settings


@pytest.mark.asyncio
async def test_search_tavily_success():
    settings.tavily_api_key = "fake_key"
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "Test Title", "url": "https://test.com", "content": "Test Content"}
        ]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await search_tavily("test query")
        lines = result.splitlines()
        assert len(lines) >= 3
        assert lines[0] == "Source: https://test.com"
        assert lines[1] == "Title: Test Title"
        assert lines[2] == "Snippet: Test Content"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_search_tavily_missing_key():
    original_key = settings.tavily_api_key
    settings.tavily_api_key = None
    try:
        with pytest.raises(ValueError, match="Tavily API key is not configured"):
            await search_tavily("test")
    finally:
        settings.tavily_api_key = original_key


@pytest.mark.asyncio
async def test_unified_search_fallback():
    settings.tavily_api_key = "fake_key"
    
    # We mock search_tavily to raise an exception, and search_duckduckgo to return mock DDG results
    with patch("app.services.search_service.search_tavily", side_effect=Exception("Tavily Error")), \
         patch("app.services.search_service.search_duckduckgo") as mock_ddg:
        
        mock_ddg.return_value = "DuckDuckGo: Result Found"
        
        result = await unified_web_search("fallback test")
        assert result == "DuckDuckGo: Result Found"
        mock_ddg.assert_called_once_with("fallback test", max_results=5)

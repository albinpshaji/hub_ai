from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai_router"}

def test_embed_endpoint(mocker):
    # Mock the OllamaClient.generate_embedding to return a dummy vector
    mocker.patch("app.main.ollama_client.generate_embedding", return_value=[0.1, 0.2, 0.3])
    
    response = client.post("/api/v1/embed", json={"text": "Hello world"})
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert response.json()["embedding"] == [0.1, 0.2, 0.3]

def test_chat_stream_endpoint(mocker):
    # Mock the RoutingEngine.process_stream to yield fake tokens
    async def fake_stream(*args, **kwargs):
        yield "Hello"
        yield " World"
        
    mocker.patch("app.main.router_engine.process_stream", side_effect=fake_stream)
    
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "user_id": "test_user",
        "use_rag": False
    }
    
    # We use stream=True in requests (which TestClient uses under the hood)
    with client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        lines = list(response.iter_lines())
        # lines will be:
        # data: {"delta": "Hello"}
        # <empty line>
        # data: {"delta": " World"}
        # <empty line>
        # data: [DONE]
        # <empty line>
        
        # Filter out empty lines
        data_lines = [line for line in lines if line]
        
        assert len(data_lines) == 3
        assert data_lines[0] == 'data: {"delta": "Hello"}'
        assert data_lines[1] == 'data: {"delta": " World"}'
        assert data_lines[2] == 'data: [DONE]'

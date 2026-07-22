import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_disabled_by_default(client):
    """When settings.ai_api_key is None, API requests succeed without auth header."""
    settings.ai_api_key = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check_bypasses_auth(client):
    """Health check and root endpoints remain accessible even when AI_API_KEY is set."""
    settings.ai_api_key = "secret_key_123"
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        response = client.get("/")
        assert response.status_code == 200
    finally:
        settings.ai_api_key = None


def test_api_key_required_when_configured(client):
    """When settings.ai_api_key is set, requests without valid X-AI-API-Key are rejected with 401."""
    settings.ai_api_key = "secret_key_123"
    try:
        # Request missing X-AI-API-Key header
        response = client.post("/api/v1/chat/summarize", json={"text": "Hello world"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"

        # Request with wrong X-AI-API-Key header
        response = client.post(
            "/api/v1/chat/summarize",
            json={"text": "Hello world"},
            headers={"X-AI-API-Key": "wrong_key"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"
    finally:
        settings.ai_api_key = None


def test_valid_api_key_accepted(client):
    """When settings.ai_api_key is set and correct header is passed, request proceeds past auth check."""
    settings.ai_api_key = "secret_key_123"
    try:
        # Pass correct key header (returns 400 or 503 or handled error from business logic, not 401)
        response = client.post(
            "/api/v1/chat/summarize",
            json={"text": "Hello world"},
            headers={"X-AI-API-Key": "secret_key_123"},
        )
        assert response.status_code != 401
    finally:
        settings.ai_api_key = None

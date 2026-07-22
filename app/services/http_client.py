"""
Shared httpx.AsyncClient singleton for the AI service.

All services should import `get_http_client()` instead of creating their own
`async with httpx.AsyncClient(...) as client:` blocks. This avoids:
  - TCP connection churn against Ollama / external APIs
  - Potential "too many open files" crashes under load
  - Unnecessary TLS/TCP handshake overhead per request

Lifecycle is managed by main.py's lifespan context manager.
"""
from __future__ import annotations

import httpx
import logging

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def init_http_client() -> None:
    """Create the shared client. Called once during app startup."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,   # Vision / LLM inference can be slow
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=120,
            ),
        )
        logger.info("Shared httpx.AsyncClient initialised.")


async def close_http_client() -> None:
    """Gracefully close the shared client. Called during app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Shared httpx.AsyncClient closed.")


def get_http_client() -> httpx.AsyncClient:
    """
    Return the shared AsyncClient instance.

    Raises RuntimeError if called before init_http_client().
    """
    if _client is None:
        raise RuntimeError(
            "Shared HTTP client is not initialised. "
            "Ensure init_http_client() is called during app startup."
        )
    return _client

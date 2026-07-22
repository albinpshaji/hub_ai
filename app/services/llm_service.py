"""
LLM service — Ollama streaming and text generation.

Provides:
  - chat_stream()    — Async generator that yields individual tokens from Ollama /api/chat
  - summarize_text() — One-shot text summarization via Ollama /api/generate
  - get_embedding()  — Convenience wrapper around vector_service embeddings
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from app.config import settings
from app.services.http_client import get_http_client
from app.services.vector_service import get_ollama_embedding  # re-export convenience

logger = logging.getLogger(__name__)


async def chat_stream(
    messages: list[dict],
    think: bool = True,
) -> AsyncIterator[str]:
    """
    Stream tokens from the local Ollama /api/chat endpoint.

    Accepts a fully-assembled message list (system prompt + chat history already
    included by the caller).  Yields individual token strings.

    Raises httpx.ConnectError if Ollama is not reachable.
    """
    client = get_http_client()
    async with client.stream(
        "POST",
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "num_ctx": 16384,
                "num_predict": -1,
            },
            "keep_alive": "5m",
        },
    ) as response:
            response.raise_for_status()
            in_thinking = False
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    message = parsed.get("message", {})
                    
                    thinking_token = message.get("thinking", "")
                    content_token = message.get("content", "")
                    
                    if thinking_token:
                        if not in_thinking:
                            in_thinking = True
                            yield "<think>"
                        yield thinking_token
                    elif content_token:
                        if in_thinking:
                            in_thinking = False
                            yield "</think>"
                        yield content_token
                except (json.JSONDecodeError, KeyError):
                    continue
            
            if in_thinking:
                yield "</think>"


async def summarize_text(text: str) -> str:
    """
    Summarize text using Ollama /api/generate (non-streaming).

    Returns a concise 3-sentence summary of the input text.
    Raises httpx.ConnectError if Ollama is not reachable.
    """
    client = get_http_client()
    resp = await client.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": (
                "Summarize the following conversation in 3 concise sentences. "
                "Focus on the main topics and key conclusions discussed:\n\n"
                f"{text}"
            ),
            "stream": False,
        },
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


async def get_embedding(text: str) -> list[float]:
    """
    Convenience wrapper — returns a 768-dim embedding from nomic-embed-text.
    Delegates to vector_service.get_ollama_embedding().
    """
    return await get_ollama_embedding(text)


async def chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    think: bool = True,
) -> dict:
    """
    One-shot chat completion with Ollama supporting tool/function calling.
    Returns the message dictionary (which may contain 'content' or 'tool_calls').
    """
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "num_ctx": 16384,
            "num_predict": -1,
            "temperature": 0.0 if tools else 0.7, # Enforce deterministic output to prevent XML drift when tools are active
        },
        "keep_alive": "5m",
    }
    if tools:
        payload["tools"] = tools

    client = get_http_client()
    try:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json().get("message", {})
    except Exception as exc:
        logger.error("Ollama chat_with_tools request failed: %s", exc)
        raise exc

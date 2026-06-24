"""
Embedding Generation — Phase 4

Connects to Ollama and generates embeddings for text chunks.
Supports concurrent batching.
"""

import asyncio
import logging
from typing import List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string using Ollama.
    """
    url = f"{settings.ollama_base_url}/api/embeddings"
    payload = {
        "model": settings.ollama_embed_model,
        "prompt": text
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except httpx.HTTPError as e:
            logger.error("HTTP error during embedding generation: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            raise

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of strings concurrently.
    Respects the embed_concurrency setting.
    """
    semaphore = asyncio.Semaphore(settings.embed_concurrency)
    
    async def _embed_with_semaphore(text: str) -> List[float]:
        async with semaphore:
            return await generate_embedding(text)
            
    tasks = [_embed_with_semaphore(text) for text in texts]
    return await asyncio.gather(*tasks)

"""
Chat Router — Phase 7

Handles RAG-augmented chat with SSE streaming from Ollama.
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from app.schemas.models import ChatRequest
from app.config import settings
from app.rag.retriever import retrieve_chunks
from app.rag.context_assembler import assemble_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

async def ollama_stream_generator(
    messages: list[dict],
    model: str = settings.ollama_chat_model
) -> AsyncGenerator[str, None]:
    """
    Calls Ollama chat endpoint and yields SSE-formatted chunks.
    """
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            # Format for SSE (Server-Sent Events)
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode Ollama stream line: %s", line)
        except httpx.HTTPStatusError as e:
            error_body = e.response.text if not hasattr(e.response, "aread") else "See logs"
            logger.error("Ollama HTTP error: %s - Body: %s", e, error_body)
            yield f"data: {json.dumps({'error': f'{str(e)} - {error_body}'})}\n\n"
        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    yield "data: [DONE]\n\n"

@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """
    Stream a chat response. If use_rag is true, augment the last message 
    with retrieved context from ChromaDB.
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
        
    messages_payload = [{"role": m.role, "content": m.content} for m in body.messages]
    
    if body.use_rag:
        # Get the latest user message to query context
        last_message = messages_payload[-1]
        if last_message["role"] == "user":
            query = last_message["content"]
            
            try:
                # Retrieve relevant chunks
                chunks = await retrieve_chunks(query, body.user_id)

                logger.info("=" * 80)
                logger.info("USER QUERY")
                logger.info(query)

                logger.info("=" * 80)
                logger.info("RETRIEVED CHUNKS")

                for i, chunk in enumerate(chunks, 1):
                    logger.info(f"Chunk {i}")
                    logger.info(chunk)
                    logger.info("-" * 60)
                if chunks:
                    context_block = assemble_context(chunks)
                    # Augment the last message content
                    augmented_content = (
                        f"Use the following context to answer the user's question. "
                        f"If the context does not contain the answer, say 'I do not have enough information to answer that based on the provided documents.' Do not hallucinate.\n\n"
                        f"Context:\n{context_block}\n\n"
                        f"User Question: {query}"
                    )
                    last_message["content"] = augmented_content
                    logger.info("Augmented chat prompt with %d chunks of context", len(chunks))
                else:
                    augmented_content = (
                        f"The user asked a question, but no relevant documents were found in their database. "
                        f"Please politely inform the user that you do not have any provided context or information about this topic to answer their question. Do not hallucinate an answer.\n\n"
                        f"User Question: {query}"
                    )
                    last_message["content"] = augmented_content
                    logger.info("No chunks found for RAG, added fallback anti-hallucination prompt.")
            except Exception as e:
                logger.error("Failed to retrieve context for chat: %s", e)
                # Fall back to normal chat if retrieval fails

    return StreamingResponse(
        ollama_stream_generator(messages_payload),
        media_type="text/event-stream"
    )

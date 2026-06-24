from fastapi import APIRouter, HTTPException
import logging

from app.rag.embedder import generate_embedding
from app.schemas.models import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["embedding"])

@router.post("/embed", response_model=EmbedResponse)
async def embed_text(body: EmbedRequest):
    """
    Generate an embedding vector for the provided text.
    Called by the backend proxy (llm_service.py).
    """
    try:
        embedding = await generate_embedding(body.text)
        return EmbedResponse(embedding=embedding)
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {type(exc).__name__}: {exc}",
        )

"""
RAG Router — Phase 5

Handles ingestion into ChromaDB and document deletion.
"""

from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.models import IngestRequest, IngestResponse, RetrieveRequest, RetrieveResponse
from app.rag.chunker import chunk_text
from app.rag.embedder import generate_embeddings_batch
from app.rag.vector_store import store_chunks, delete_document
from app.rag.retriever import retrieve_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(body: IngestRequest):
    """
    Ingest a document into the RAG pipeline.
    Chunks the text, embeds chunks via Ollama, and stores them in ChromaDB.
    """
    try:
        # 1. Chunking
        metadata = {
            "document_id": body.document_id,
            "user_id": body.user_id,
            "filename": body.filename
        }
        chunks = chunk_text(body.text, metadata)
        
        if not chunks:
            return IngestResponse(chunks_stored=0)
            
        # 2. Embedding
        texts_to_embed = [c.text for c in chunks]
        embeddings = await generate_embeddings_batch(texts_to_embed)
        
        # 3. Storage
        chunk_ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        # ChromaDB does not allow None values in metadata
        metadatas = [c.metadata.model_dump(exclude_none=True) for c in chunks]
        
        # Run synchronous ChromaDB operation in a separate thread if needed,
        # but for simplicity calling it directly here.
        store_chunks(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return IngestResponse(chunks_stored=len(chunks))
        
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {type(exc).__name__}: {exc}"
        )

@router.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: str, user_id: str = Query(..., description="User ID for authorization")):
    """
    Delete a document's chunks from ChromaDB.
    """
    try:
        delete_document(document_id, user_id)
        return {"status": "success"}
    except Exception as exc:
        logger.error("Deletion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Deletion failed: {type(exc).__name__}: {exc}"
        )

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(body: RetrieveRequest):
    """
    Retrieve the most relevant document chunks for a given query.
    """
    try:
        chunks = await retrieve_chunks(
            query=body.query,
            user_id=body.user_id,
            top_k=body.top_k
        )
        return RetrieveResponse(chunks=chunks)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {type(exc).__name__}: {exc}"
        )

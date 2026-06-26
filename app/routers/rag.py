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
from app.rag.extractor import extract_text


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(body: IngestRequest):
    """
    Complete RAG ingestion pipeline.

    Upload
      ↓
    Extract
      ↓
    Chunk
      ↓
    Embed
      ↓
    Store in ChromaDB
    """

    try:

        logger.info(
            "Starting ingestion for document %s",
            body.document_id,
        )

        # ----------------------------------------------------
        # STEP 1 : Extract text
        # ----------------------------------------------------

        extraction = extract_text(
            body.file_path,
            body.file_type,
        )

        logger.info(
            "Extracted %d characters",
            len(extraction.raw_text),
        )

        # ----------------------------------------------------
        # STEP 2 : Chunk
        # ----------------------------------------------------

        metadata = {
            "document_id": body.document_id,
            "user_id": body.user_id,
            "filename": body.filename,
        }

        chunks = chunk_text(
            extraction.raw_text,
            metadata,
        )

        if not chunks:
            return IngestResponse(
                chunks_stored=0,
            )

        logger.info(
            "Generated %d chunks",
            len(chunks),
        )

        # ----------------------------------------------------
        # STEP 3 : Generate embeddings
        # ----------------------------------------------------

        texts = [chunk.text for chunk in chunks]

        embeddings = await generate_embeddings_batch(texts)

        # ----------------------------------------------------
        # STEP 4 : Store in ChromaDB
        # ----------------------------------------------------

        logger.info(
            "Storing %d chunks for user %s",
            len(chunks),
            body.user_id,
        )
        

        store_chunks(
            chunk_ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                c.metadata.model_dump(exclude_none=True)
                for c in chunks
            ],
        )

        logger.info(
            "Stored %d chunks successfully",
            len(chunks),
        )

        return IngestResponse(
            chunks_stored=len(chunks),
        )

    except Exception as exc:

        logger.exception(exc)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
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

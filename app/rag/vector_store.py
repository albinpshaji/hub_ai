"""
Vector Store — Phase 5

Connects to ChromaDB to store and manage embedded document chunks.
"""

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings

logger = logging.getLogger(__name__)

_client = None
_collection = None

def init_chroma():
    """Initialize ChromaDB client and get/create collection."""
    global _client, _collection
    try:
        # Using synchronous HttpClient. For a high-throughput async app,
        # AsyncHttpClient could be used, but HttpClient is fine for this phase.
        _client = chromadb.HttpClient(
            host=settings.chroma_host, 
            port=settings.chroma_port
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized, collection '%s' ready.", settings.chroma_collection)
    except Exception as e:
        logger.error("Failed to initialize ChromaDB: %s", e)
        # We don't raise here so the app can still boot if Chroma is down,
        # but subsequent get_collection() calls will fail.

def get_collection() -> Collection:
    """Get the initialized ChromaDB collection."""
    if _collection is None:
        raise RuntimeError("ChromaDB collection is not initialized. Is ChromaDB running?")
    return _collection

def store_chunks(
    chunk_ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]]
):
    """Store embedded chunks in ChromaDB."""
    if not chunk_ids:
        return
    
    print("=" * 60)
    print("STORING CHUNKS")
    print("First metadata:")
    print(metadatas[0])
    print("=" * 60)
            
    collection = get_collection()
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    logger.info("Stored %d chunks in ChromaDB", len(chunk_ids))

def delete_document(document_id: str, user_id: str):
    """Delete all chunks for a specific document belonging to a user."""
    collection = get_collection()
    
    collection.delete(
        where={
            "$and": [
                {"document_id": document_id},
                {"user_id": user_id}
            ]
        }
    )
    logger.info("Deleted document %s for user %s from ChromaDB", document_id, user_id)

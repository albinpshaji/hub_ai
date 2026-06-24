"""
Retriever — Phase 6

Queries ChromaDB using embedded user queries.
"""

import logging
from typing import List

from app.rag.embedder import generate_embedding
from app.rag.vector_store import get_collection

logger = logging.getLogger(__name__)

async def retrieve_chunks(query: str, user_id: str, top_k: int = 5) -> List[str]:
    """
    Embed the user query, search ChromaDB, filter by user, 
    and return the top matching text chunks.
    """
    if not query.strip():
        return []
        
    try:
        # 1. Embed the query
        query_embedding = await generate_embedding(query)
        
        # 2. Search ChromaDB
        collection = get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        
        # results["documents"] is a list of lists: [['chunk1', 'chunk2', ...]]
        documents = results.get("documents", [])
        if not documents or not documents[0]:
            return []
            
        # Extract the documents from the first (and only) query result
        retrieved_chunks = documents[0]
        
        # Deduplicate while preserving order
        seen = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            # Avoid completely identical chunks
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
                
        logger.info("Retrieved %d unique chunks for user %s", len(unique_chunks), user_id)
        return unique_chunks
        
    except Exception as e:
        logger.error("Failed to retrieve chunks: %s", e)
        raise

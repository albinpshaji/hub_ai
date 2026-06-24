"""
Semantic Chunking — Phase 3

Splits extracted text into ~500-token chunks with 50-token overlap using a
hybrid strategy (structure-aware + recursive fallback).
"""

import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.models import ChunkResult, ChunkMetadata

logger = logging.getLogger(__name__)

# Using an approximation: 1 token ≈ 4 characters
# Target: ~500 tokens (2000 chars), overlap: 50 tokens (200 chars)
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

def chunk_text(text: str, metadata: dict[str, Any]) -> list[ChunkResult]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The full text of the document or page.
        metadata: A dictionary containing at least 'document_id', 'user_id', 
                  'filename', and optionally 'page_number', 'section_heading'.
                  
    Returns:
        List of ChunkResult objects.
    """
    if not text or not text.strip():
        return []

    # Initialize Langchain's recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", " ", ""]
    )
    
    text_chunks = splitter.split_text(text)
    
    document_id = metadata.get("document_id", "unknown_doc")
    user_id = metadata.get("user_id", "unknown_user")
    filename = metadata.get("filename", "unknown.txt")
    page_number = metadata.get("page_number", 1)
    section_heading = metadata.get("section_heading")
    
    results = []
    for i, chunk_txt in enumerate(text_chunks):
        # Approximate token count for metadata
        token_count = max(1, len(chunk_txt) // 4)
        
        chunk_meta = ChunkMetadata(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            page_number=page_number,
            section_heading=section_heading,
            token_count=token_count
        )
        
        result = ChunkResult(
            chunk_id=f"{document_id}_chunk_{i}",
            text=chunk_txt,
            chunk_index=i,
            metadata=chunk_meta
        )
        results.append(result)
        
    logger.info("Split text into %d chunks for doc %s", len(results), document_id)
    return results

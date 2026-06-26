"""
Pydantic request/response models for the AI service.

These schemas define the API contract that the hub_backend proxy layer expects.
Each schema is documented with which backend proxy service calls it.
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Phase 1 — Extraction
# Called by: hub_backend/app/services/document_service.py
# ═══════════════════════════════════════════════════════════════


class ExtractRequest(BaseModel):
    """
    Request from backend's document_service.extract_text().

    file_path: Absolute path to the saved file on disk.
               In dev, this is inside hub_backend's uploads/ directory.
               Both services must have access to the same filesystem.

    file_type: Extension without the dot — "pdf", "docx", "txt", "png", "jpg"
    """

    file_path: str = Field(..., description="Absolute path to the saved file")
    file_type: str = Field(..., description="File extension without dot: pdf, docx, txt")


class DocumentMetadata(BaseModel):
    """
    Rich metadata extracted from the document.
    """
    author: Optional[str] = Field(None, description="Document author")
    title: Optional[str] = Field(None, description="Document title")
    creation_date: Optional[str] = Field(None, description="Creation date string")
    total_chars: int = Field(0, description="Total number of characters in the document")
    extraction_method: str = Field("text", description="Extraction method used")


class ExtractResponse(BaseModel):
    """
    Response returned to backend's document_service.extract_text().

    The backend expects at minimum: { "text": "..." }
    We return extra fields for downstream pipeline use.
    """

    text: str = Field(..., description="Extracted plain text from the document")
    page_count: int = Field(0, description="Number of pages (0 for TXT)")
    extraction_method: str = Field(
        "text", description="How text was extracted: 'text', 'ocr', or 'hybrid'"
    )
    metadata: Optional[DocumentMetadata] = Field(
        None, description="Extracted metadata from the document"
    )

# ═══════════════════════════════════════════════════════════════
# Phase 3 — Chunking
# ═══════════════════════════════════════════════════════════════


class ChunkMetadata(BaseModel):
    document_id: str = Field(..., description="UUID of the parent document")
    user_id: str = Field(..., description="UUID of the user who owns the document")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(1, description="Page number where this chunk starts")
    section_heading: Optional[str] = Field(None, description="Heading under which this chunk falls")
    token_count: int = Field(..., description="Number of tokens in the chunk")


class ChunkResult(BaseModel):
    chunk_id: str = Field(..., description="Unique ID for this chunk, e.g. doc123_chunk_0")
    text: str = Field(..., description="The chunk text itself")
    chunk_index: int = Field(..., description="0-based index of this chunk in the document")
    metadata: ChunkMetadata

# ═══════════════════════════════════════════════════════════════
# Phase 4 — Embedding
# ═══════════════════════════════════════════════════════════════

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed")

class EmbedResponse(BaseModel):
    embedding: list[float] = Field(..., description="768-dimensional float array representing the text")

# ═══════════════════════════════════════════════════════════════
# Phase 5 — Ingestion
# ═══════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    """
    Request sent by hub_notify after a document upload.

    The AI service is responsible for:
    - Extracting text
    - Chunking
    - Embedding
    - Storing vectors
    """

    user_id: str = Field(..., description="UUID of the user")

    document_id: str = Field(..., description="UUID of the document")

    file_path: str = Field(
        ...,
        description="Absolute path of the uploaded document"
    )

    file_type: str = Field(
        ...,
        description="Document type (pdf, docx, txt...)"
    )

    filename: str = Field(
        ...,
        description="Original filename"
    )

class IngestResponse(BaseModel):
    chunks_stored: int = Field(..., description="Number of chunks generated and stored")

# ═══════════════════════════════════════════════════════════════
# Phase 6 — Retrieval
# ═══════════════════════════════════════════════════════════════

class RetrieveRequest(BaseModel):
    user_id: str = Field(..., description="UUID of the user")
    query: str = Field(..., description="The user's query string")
    top_k: int = Field(5, description="Number of chunks to retrieve")

class RetrieveResponse(BaseModel):
    chunks: list[str] = Field(..., description="List of relevant text chunks")

# ═══════════════════════════════════════════════════════════════
# Phase 7 — Chat
# ═══════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")

class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Conversation history")
    user_id: str = Field(..., description="UUID of the user")
    use_rag: bool = Field(True, description="Whether to use RAG to augment the response")




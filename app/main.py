"""
CixioHub AI Service — FastAPI Application Entry Point

This is the "missing" AI service that hub_backend expects at port 8003.
The backend's proxy services (llm_service.py, rag_service.py, document_service.py)
all make HTTP calls to this service.

Phase 1: /api/v1/health + /api/v1/extract
Phase 4: /api/v1/embed
Phase 5: /api/v1/rag/ingest, /api/v1/rag/documents/{id}
Phase 6: /api/v1/rag/retrieve
Phase 7: /api/v1/chat/stream

Start with: uvicorn app.main:app --reload --port 8003
Swagger UI: http://localhost:8003/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import extract_router, embed_router, rag_router, chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs on startup and shutdown.

    Future phases will initialize ChromaDB connection here.
    """
    # ── Startup ──────────────────────────────────
    # Phase 5: Initialize ChromaDB client/collection here
    from app.rag.vector_store import init_chroma
    init_chroma()
    yield
    # ── Shutdown ─────────────────────────────────
    # Cleanup resources if needed


app = FastAPI(
    title="CixioHub AI Service",
    version="1.0.0",
    description=(
        "AI service for CixioHub — handles document extraction, RAG pipeline, "
        "embedding generation, and LLM chat streaming."
    ),
    lifespan=lifespan,
)

# CORS — allow the backend and frontend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:8000",  # FastAPI backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers under /api/v1 ──────────────────────────
PREFIX = "/api/v1"
app.include_router(extract_router, prefix=PREFIX)
app.include_router(embed_router, prefix=PREFIX)
app.include_router(rag_router, prefix=PREFIX)
app.include_router(chat_router, prefix=PREFIX)


# ── Health check (always available) ──────────────────────────
@app.get("/api/v1/health", tags=["health"])
async def health():
    """Health check endpoint. Returns OK if the service is running."""
    return {"status": "ok", "service": "cixiohub-ai"}

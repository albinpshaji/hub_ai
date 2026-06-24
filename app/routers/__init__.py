from app.routers.extract import router as extract_router
from app.routers.embed import router as embed_router
from app.routers.rag import router as rag_router
from app.routers.chat import router as chat_router

__all__ = ["extract_router", "embed_router", "rag_router", "chat_router"]

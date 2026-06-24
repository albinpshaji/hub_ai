import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.router.engine import RoutingEngine
from app.core.llm import OllamaClient

app = FastAPI(
    title="CixioHub AI Router API",
    description="Intelligent LLM router and embeddings API for CixioHub.",
    version="1.0.0"
)

router_engine = RoutingEngine()
ollama_client = OllamaClient()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    use_rag: Optional[bool] = False

class EmbedRequest(BaseModel):
    text: str

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok", "service": "ai_router"}

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Accepts a chat request and streams the response back using Server-Sent Events (SSE).
    """
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]

    async def sse_generator():
        try:
            async for token in router_engine.process_stream(messages_dict):
                # Format as SSE required by the backend
                payload = json.dumps({"delta": token})
                yield f"data: {payload}\n\n"
        except Exception as e:
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.post("/api/v1/embed")
async def generate_embeddings(request: EmbedRequest):
    """
    Generates vector embeddings for the provided text using the AI engine.
    """
    embedding = await ollama_client.generate_embedding(request.text)
    return {"embedding": embedding}
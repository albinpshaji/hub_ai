import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, status, Query, UploadFile, File, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.config import settings
from app.schemas import (
    ChatStreamRequest,
    SummarizeRequest,
    EmbeddingRequest,
    IngestRequest,
    ChatWithToolsRequest,
    IngestImagesRequest,
    SearchRequest,
    ExtractRequest,
    WebSearchRequest,
    ExtractVisualsRequest,
    ReinspectPageRequest,
    VisionQARequest,
)
from app.services.llm_service import chat_stream, summarize_text, get_embedding, chat_with_tools
from app.services.vector_service import (
    init_qdrant_collection,
    store_document_vectors,
    store_image_vectors,
    search_relevant_chunks,
    delete_document_vectors,
)
from app.services.document_service import extract_text
from app.services.search_service import unified_web_search
from app.services.vision_service import (
    extract_visuals_from_pdf,
    reinspect_page,
    process_and_compress_image,
)
from app.services.http_client import init_http_client, close_http_client, get_http_client
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_api_key(request: Request):
    if settings.ai_api_key:
        key = request.headers.get("X-AI-API-Key")
        if key != settings.ai_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )


def _safe_error(exc: Exception, context: str = "AI service") -> tuple[int, str]:
    """
    Map exceptions to safe HTTP status codes and sanitized messages.
    Prevents leaking internal paths, stack traces, or Ollama errors to clients.
    """
    if isinstance(exc, httpx.ConnectError):
        return 503, f"{context}: model service is unavailable"
    if isinstance(exc, httpx.TimeoutException):
        return 504, f"{context}: request timed out"
    if isinstance(exc, httpx.HTTPStatusError):
        return 502, f"{context}: upstream service returned an error"
    if isinstance(exc, ValueError):
        return 400, f"{context}: {exc}"
    # Generic fallback — never expose raw str(exc)
    logger.error("Unhandled error in %s: %s", context, exc, exc_info=True)
    return 500, f"{context}: internal error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared HTTP client and vector collection at startup
    await init_http_client()
    try:
        await init_qdrant_collection()
    except Exception as e:
        logger.error("Failed to initialize Qdrant collection: %s", e)
    yield
    # Gracefully close shared HTTP client on shutdown
    await close_http_client()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="CixioHub Standalone AI/ML Microservice",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])



@router.post("/chat/stream")
async def handle_chat_stream(payload: ChatStreamRequest):
    async def event_generator():
        in_thinking = False
        try:
            async for token in chat_stream(payload.messages, think=payload.think):
                if token == "<think>":
                    in_thinking = True
                    continue
                elif token == "</think>":
                    in_thinking = False
                    continue

                if in_thinking:
                    data = {"thinking": token}
                else:
                    data = {"delta": token}
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.error("Error generating stream: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/summarize")
async def handle_summarize(payload: SummarizeRequest):
    try:
        summary = await summarize_text(payload.text)
        return {"summary": summary}
    except Exception as exc:
        code, msg = _safe_error(exc, "Summarization")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/embeddings")
async def handle_embeddings(payload: EmbeddingRequest):
    try:
        embedding = await get_embedding(payload.text)
        return {"embedding": embedding}
    except Exception as exc:
        code, msg = _safe_error(exc, "Embedding")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/documents/ingest")
async def handle_ingest(payload: IngestRequest):
    try:
        chunks_stored = await store_document_vectors(
            user_id=payload.user_id,
            document_id=payload.document_id,
            text=payload.text,
            filename=payload.filename,
            session_id=payload.session_id,
        )
        return {"chunks_stored": chunks_stored}
    except Exception as exc:
        code, msg = _safe_error(exc, "Document ingestion")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/chat/tools")
async def handle_chat_with_tools(payload: ChatWithToolsRequest):
    try:
        message = await chat_with_tools(
            messages=payload.messages,
            tools=payload.tools,
            think=payload.think,
        )
        return {"message": message}
    except Exception as exc:
        code, msg = _safe_error(exc, "Tool-calling chat")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/documents/ingest_images")
async def handle_ingest_images(payload: IngestImagesRequest):
    try:
        chunks_stored = await store_image_vectors(
            user_id=payload.user_id,
            document_id=payload.document_id,
            filename=payload.filename,
            image_metadata=payload.image_metadata,
            session_id=payload.session_id,
        )
        return {"chunks_stored": chunks_stored}
    except Exception as exc:
        code, msg = _safe_error(exc, "Image ingestion")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/documents/search")
async def handle_search(payload: SearchRequest):
    try:
        results = await search_relevant_chunks(
            user_id=payload.user_id,
            query=payload.query,
            limit=payload.limit,
            retrieval_mode=payload.retrieval_mode,
            use_hyde=payload.use_hyde,
            allowed_document_ids=payload.allowed_document_ids,
            session_id=payload.session_id,
            selected_document_ids=payload.selected_document_ids,
            use_reranker=payload.use_reranker,
            include_meta=payload.include_meta,
        )
        return {"results": results}
    except Exception as exc:
        code, msg = _safe_error(exc, "Document search")
        raise HTTPException(status_code=code, detail=msg)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def handle_delete_document(document_id: str, user_id: str = Query(...)):
    try:
        import uuid
        await delete_document_vectors(
            user_id=uuid.UUID(user_id),
            document_id=uuid.UUID(document_id),
        )
    except Exception as exc:
        code, msg = _safe_error(exc, "Document deletion")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/extract")
async def handle_extract(payload: ExtractRequest):
    try:
        text = await extract_text(
            file_path=payload.file_path,
            file_type=payload.file_type,
        )
        return {"text": text}
    except Exception as exc:
        code, msg = _safe_error(exc, "Text extraction")
        raise HTTPException(status_code=code, detail=msg)


@app.get("/")
async def root():
    return {"status": "ok", "service": settings.app_name}


@app.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/web/search")
async def handle_web_search(payload: WebSearchRequest):
    try:
        result = await unified_web_search(payload.query, max_results=payload.max_results)
        return {"result": result}
    except Exception as exc:
        code, msg = _safe_error(exc, "Web search")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/vision/compress")
async def handle_vision_compress(file: UploadFile = File(...)):
    try:
        content = await file.read()
        compressed = process_and_compress_image(content)
        b64_str = base64.b64encode(compressed).decode("utf-8")
        return {"base64_image": b64_str}
    except Exception as exc:
        code, msg = _safe_error(exc, "Image compression")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/vision/extract_visuals")
async def handle_extract_visuals(payload: ExtractVisualsRequest):
    try:
        visuals = extract_visuals_from_pdf(payload.pdf_path)
        return {"visuals": visuals}
    except Exception as exc:
        code, msg = _safe_error(exc, "Visual extraction")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/vision/reinspect")
async def handle_reinspect(payload: ReinspectPageRequest):
    try:
        desc = await reinspect_page(
            pdf_path=payload.pdf_path,
            page_number=payload.page_number,
            specific_question=payload.specific_question,
        )
        return {"description": desc}
    except Exception as exc:
        code, msg = _safe_error(exc, "Page reinspection")
        raise HTTPException(status_code=code, detail=msg)


@router.post("/vision/qa")
async def handle_vision_qa(payload: VisionQARequest):
    """Answer a specific question about an image using the vision model.

    Previous implementation called describe_image() first (wasting a full
    GPU inference pass) then immediately discarded the result and ran a
    second inference with the user's question. Now we go straight to the
    targeted question — halving GPU work per request.
    """
    try:
        prompt = (
            f"Look at this image. Answer the following question based on the visual contents: {payload.question}"
        )
        client = get_http_client()
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_vision_model,
                "prompt": prompt,
                "images": [payload.base64_image],
                "stream": False,
                "think": False,
                "options": {
                    "num_ctx": 4096,
                },
                "keep_alive": "10s",
            }
        )
        response.raise_for_status()
        res_data = response.json()
        desc = res_data.get("response", "").strip()
        if not desc:
            desc = res_data.get("thinking", "").strip()
        return {"description": desc}
    except Exception as exc:
        code, msg = _safe_error(exc, "Vision QA")
        raise HTTPException(status_code=code, detail=msg)


app.include_router(router)

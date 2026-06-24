"""
Extract router — POST /api/v1/extract

This endpoint is called by hub_backend's document_service.py proxy.
It receives a file path and file type, extracts the text, and returns it.

Flow:
  hub_backend (background task after file upload)
    → POST /api/v1/extract { file_path, file_type }
    ← { text, page_count, extraction_method }
"""

from fastapi import APIRouter, HTTPException

from app.rag.extractor import extract_text
from app.schemas.models import ExtractRequest, ExtractResponse

router = APIRouter(tags=["extraction"])


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(body: ExtractRequest):
    """
    Extract plain text from a document file.

    Supports: PDF (with OCR fallback), DOCX, TXT.

    The backend calls this after saving an uploaded file to disk.
    The file_path must be accessible from this service's filesystem.

    Returns the extracted text plus metadata about the extraction.
    """
    try:
        result = extract_text(body.file_path, body.file_type)

        return ExtractResponse(
            text=result.raw_text,
            page_count=result.page_count,
            extraction_method=result.extraction_method,
            metadata=result.metadata,
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {type(exc).__name__}: {exc}",
        )

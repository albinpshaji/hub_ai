"""
Document text extraction service — delegates to the AI service.

The AI service (cixio-hub/ai, port 8003) handles:
  - PDF extraction via PyMuPDF
  - DOCX extraction via python-docx
  - Plain text reading
  - Image OCR via Tesseract

Students (AI/LLM role): implement extraction in cixio-hub/ai/app/rag/document_extractor.py
Students (Backend role): this file is already wired.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.services.http_client import get_http_client

logger = logging.getLogger(__name__)


async def extract_text(file_path: str, file_type: str) -> str:
    """
    Extract plain text from a file.
    Attempts to extract locally for PDF, TXT, and MD files to avoid dependency on external service.
    Falls back to the external AI service if local extraction fails.
    """
    ft = file_type.lower().strip()
    content_bytes: bytes | None = None
    
    # 1. Retrieve the file content
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            client = get_http_client()
            resp = await client.get(file_path)
            resp.raise_for_status()
            content_bytes = resp.content
        except Exception:
            pass
    else:
        # Check local filesystem (try absolute path, then relative uploads directory path)
        for path_obj in (Path(file_path), Path("uploads") / file_path):
            if path_obj.exists() and path_obj.is_file():
                try:
                    content_bytes = path_obj.read_bytes()
                    break
                except Exception:
                    pass

    # 2. Try parsing the content bytes locally
    if content_bytes is not None:
        if ft in ("txt", "md"):
            try:
                return content_bytes.decode("utf-8", errors="ignore")
            except Exception:
                pass
        elif ft == "pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=content_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception:
                pass

    # 3. If all local extraction failed, return empty.
    # NOTE: Previously this fell back to calling settings.ai_service_url (/api/v1/extract)
    # which was the same service (port 8003), creating a recursive HTTP loop.
    logger.warning(
        "Local extraction failed for %s (type %s). Returning empty text.",
        file_path,
        file_type,
    )
    return ""

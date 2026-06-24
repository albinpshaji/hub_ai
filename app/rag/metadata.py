"""
Metadata Extraction — Phase 2

Extracts rich metadata from PDF, DOCX, and TXT files.
"""

from pathlib import Path
from typing import Optional

from app.schemas.models import DocumentMetadata

def extract_metadata(file_path: str, file_type: str, total_chars: int, extraction_method: str) -> DocumentMetadata:
    """
    Extract metadata from a document based on its type.
    """
    path = Path(file_path)
    file_type_lower = file_type.lower().strip()
    
    author = None
    title = None
    creation_date = None
    
    if file_type_lower == "pdf":
        author, title, creation_date = _extract_pdf_metadata(path)
    elif file_type_lower == "docx":
        author, title, creation_date = _extract_docx_metadata(path)
    # TXT files don't have standard rich internal metadata

    return DocumentMetadata(
        author=author,
        title=title,
        creation_date=creation_date,
        total_chars=total_chars,
        extraction_method=extraction_method,
    )

def _parse_pdf_date(date_str: str) -> Optional[str]:
    """Parse PDF date format (D:YYYYMMDDHHmmSSZ) to standard ISO string."""
    if not date_str or not date_str.startswith("D:"):
        return date_str
    
    cleaned = date_str[2:].replace("'", "")
    if len(cleaned) >= 14:
        try:
            return f"{cleaned[0:4]}-{cleaned[4:6]}-{cleaned[6:8]}T{cleaned[8:10]}:{cleaned[10:12]}:{cleaned[12:14]}Z"
        except Exception:
            pass
    return date_str

def _extract_pdf_metadata(path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract metadata from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        metadata = doc.metadata
        doc.close()
        
        author = metadata.get("author") if metadata.get("author") else None
        title = metadata.get("title") if metadata.get("title") else None
        creation_date_raw = metadata.get("creationDate")
        
        creation_date = _parse_pdf_date(creation_date_raw) if creation_date_raw else None
        
        return author, title, creation_date
    except Exception:
        return None, None, None

def _extract_docx_metadata(path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract metadata from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(path))
        core_props = doc.core_properties
        
        author = core_props.author if core_props.author else None
        title = core_props.title if core_props.title else None
        creation_date = str(core_props.created) if core_props.created else None
        
        return author, title, creation_date
    except Exception:
        return None, None, None

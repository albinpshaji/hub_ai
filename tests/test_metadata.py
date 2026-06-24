import os
from pathlib import Path
from app.rag.metadata import extract_metadata

SAMPLE_DIR = Path(__file__).parent / "sample_files"

def test_extract_metadata_pdf():
    pdf_path = SAMPLE_DIR / "sample.pdf"
    if not pdf_path.exists():
        # Skip if sample not found
        return
    
    metadata = extract_metadata(str(pdf_path), "pdf", 1000, "text")
    assert metadata is not None
    assert metadata.extraction_method == "text"
    assert metadata.total_chars == 1000
    # Depending on the file, author/title might be None, but they should be parsed without errors
    assert hasattr(metadata, "author")
    assert hasattr(metadata, "title")
    assert hasattr(metadata, "creation_date")

def test_extract_metadata_docx():
    docx_path = SAMPLE_DIR / "sample.docx"
    if not docx_path.exists():
        return
    
    metadata = extract_metadata(str(docx_path), "docx", 500, "text")
    assert metadata is not None
    assert metadata.extraction_method == "text"
    assert metadata.total_chars == 500
    assert hasattr(metadata, "author")
    assert hasattr(metadata, "title")

def test_extract_metadata_txt():
    txt_path = SAMPLE_DIR / "sample.txt"
    if not txt_path.exists():
        return
    
    metadata = extract_metadata(str(txt_path), "txt", 200, "text")
    assert metadata is not None
    assert metadata.extraction_method == "text"
    assert metadata.total_chars == 200
    assert metadata.author is None
    assert metadata.title is None
    assert metadata.creation_date is None

import pytest
from app.rag.chunker import chunk_text

def test_chunk_empty_text():
    metadata = {"document_id": "doc123", "user_id": "u1", "filename": "test.txt"}
    chunks = chunk_text("", metadata)
    assert len(chunks) == 0

def test_chunk_short_text():
    metadata = {"document_id": "doc123", "user_id": "u1", "filename": "test.txt"}
    text = "This is a short paragraph."
    chunks = chunk_text(text, metadata)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "doc123_chunk_0"
    assert chunks[0].chunk_index == 0

def test_chunk_long_text():
    metadata = {"document_id": "doc123", "user_id": "u1", "filename": "test.txt"}
    
    # Generate text longer than 2000 chars
    sentence = "This is a sentence to test chunking. "
    text = sentence * 100  # ~3700 chars
    
    chunks = chunk_text(text, metadata)
    
    assert len(chunks) > 1
    # Check overlap (the last part of chunk 0 should be in chunk 1)
    overlap_text = chunks[0].text[-100:]
    assert overlap_text in chunks[1].text
    
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata.document_id == "doc123"

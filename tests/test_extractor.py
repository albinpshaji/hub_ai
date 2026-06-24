"""
Phase 1 Tests — Document Extraction

Tests the extractor module directly (unit tests) and the
/api/v1/extract endpoint (integration test via FastAPI TestClient).

Run with:
    cd hub_ai
    pytest tests/test_extractor.py -v
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.extractor import ExtractionResult, extract_text

# ── Paths to sample files ─────────────────────────────────────
SAMPLE_DIR = Path(__file__).parent / "sample_files"
SAMPLE_TXT = SAMPLE_DIR / "sample.txt"


# ═══════════════════════════════════════════════════════════════
# Unit Tests — extractor.py
# ═══════════════════════════════════════════════════════════════


class TestTxtExtractor:
    """Tests for plain text file extraction."""

    def test_extract_txt_returns_result(self):
        """Basic TXT extraction should return an ExtractionResult."""
        result = extract_text(str(SAMPLE_TXT), "txt")

        assert isinstance(result, ExtractionResult)
        assert len(result.raw_text) > 0
        assert result.page_count == 1
        assert result.extraction_method == "text"

    def test_extract_txt_contains_expected_content(self):
        """Extracted text should contain known content from the sample file."""
        result = extract_text(str(SAMPLE_TXT), "txt")

        assert "Machine learning" in result.raw_text
        assert "Deep learning" in result.raw_text
        assert "Natural Language Processing" in result.raw_text

    def test_extract_txt_page_info(self):
        """TXT files should have exactly 1 page with correct char count."""
        result = extract_text(str(SAMPLE_TXT), "txt")

        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1
        assert result.pages[0].char_count > 0
        assert result.pages[0].char_count == len(result.raw_text)


class TestExtractorEdgeCases:
    """Tests for error handling and edge cases."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/path/file.pdf", "pdf")

    def test_unsupported_file_type(self):
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(str(SAMPLE_TXT), "xlsx")

    def test_file_type_case_insensitive(self):
        """File type should be case-insensitive."""
        result = extract_text(str(SAMPLE_TXT), "TXT")
        assert len(result.raw_text) > 0

        result2 = extract_text(str(SAMPLE_TXT), "Txt")
        assert len(result2.raw_text) > 0


# ═══════════════════════════════════════════════════════════════
# Integration Tests — FastAPI endpoint
# ═══════════════════════════════════════════════════════════════


class TestExtractEndpoint:
    """Tests for the POST /api/v1/extract endpoint."""

    def setup_method(self):
        """Create a fresh test client for each test."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Health check should return OK."""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "cixiohub-ai"

    def test_extract_txt_via_api(self):
        """POST /api/v1/extract should extract text from a TXT file."""
        response = self.client.post(
            "/api/v1/extract",
            json={
                "file_path": str(SAMPLE_TXT),
                "file_type": "txt",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check the response matches the contract backend expects
        assert "text" in data
        assert len(data["text"]) > 0
        assert "Machine learning" in data["text"]
        assert data["page_count"] == 1
        assert data["extraction_method"] == "text"

    def test_extract_file_not_found(self):
        """Should return 404 for nonexistent files."""
        response = self.client.post(
            "/api/v1/extract",
            json={
                "file_path": "/nonexistent/file.pdf",
                "file_type": "pdf",
            },
        )
        assert response.status_code == 404

    def test_extract_unsupported_type(self):
        """Should return 400 for unsupported file types."""
        response = self.client.post(
            "/api/v1/extract",
            json={
                "file_path": str(SAMPLE_TXT),
                "file_type": "xlsx",
            },
        )
        assert response.status_code == 400

    def test_extract_missing_fields(self):
        """Should return 422 for missing required fields."""
        response = self.client.post(
            "/api/v1/extract",
            json={"file_path": str(SAMPLE_TXT)},
            # Missing file_type
        )
        assert response.status_code == 422

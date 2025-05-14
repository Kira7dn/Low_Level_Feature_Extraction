import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.constants import (
    ValidationRules, 
    validate_response_structure,
    validate_processing_time,
    get_performance_rating
)

# Constants for test configuration
MAX_PROCESSING_TIME = 5.0  # seconds
EXPECTED_KEYS = ["text"]

# Test client setup
client = TestClient(app)

def test_extract_text_success():
    """Validate successful text extraction"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'text_sample.png')
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/text/extract",
            files={"file": (f.name, f, "image/png")}
        )
    
    # Status code validation
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Processing time validation
    processing_time = response.elapsed.total_seconds()
    assert validate_processing_time(processing_time), \
        f"Processing time {processing_time}s exceeds maximum limit"
    
    # Performance rating
    performance_rating = get_performance_rating(processing_time)
    print(f"Performance Rating: {performance_rating}")
    
    # Response structure validation
    result = response.json()
    assert validate_response_structure(
        result, 
        ValidationRules.TEXT_EXTRACTION["expected_keys"],
        [
            lambda r: isinstance(r["lines"], list),
            lambda r: 0 < len(r["lines"]) <= ValidationRules.TEXT_EXTRACTION["max_text_entries"],
            lambda r: all(len(text) >= ValidationRules.TEXT_EXTRACTION["min_text_length"] for text in r["lines"])
        ]
    ), "Invalid response structure"
    assert all(isinstance(text, str) and len(text.strip()) > 0 for text in result["lines"]), \
        "All text entries must be non-empty strings"

def test_extract_text_invalid_format():
    """Test text extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/text/extract",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
    assert "Unsupported image format" in str(data["detail"]["error"]["message"])

def test_extract_text_no_file():
    """Test text extraction with no file"""
    response = client.post("/text/extract")
    
    assert response.status_code == 422  # Validation error for missing file
    data = response.json()
    assert isinstance(data['detail'], list)
    assert any(error.get('loc') == ['body', 'file'] for error in data['detail'])
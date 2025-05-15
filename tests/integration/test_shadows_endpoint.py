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

# Test client setup
client = TestClient(app)

def test_extract_shadows_success():
    """Validate successful shadows extraction"""
    test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'shadows_sample.png')
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/shadows/extract",
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
    is_valid, error_msg = validate_response_structure(
        result, 
        ValidationRules.SHADOWS_EXTRACTION["expected_keys"],
        value_types={"shadow_level": str},
        additional_checks=[
            lambda r: r["shadow_level"] in ValidationRules.SHADOWS_EXTRACTION["valid_levels"]
        ]
    )
    assert is_valid, f"Invalid response structure: {error_msg}"

def test_extract_shadows_invalid_format():
    """Test shadows extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/api/v1/shadows/extract",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]
    assert "Unsupported image format" in str(data["detail"]["error"]["message"])

def test_extract_shadows_no_file():
    """Test shadows extraction with no file"""
    response = client.post("/api/v1/shadows/extract")
    
    assert response.status_code in [400, 422]  # Either validation error or missing file error
    data = response.json()
    assert "detail" in data
    assert isinstance(data['detail'], list)
    assert any('file' in str(error.get('loc', '')) for error in data['detail'])

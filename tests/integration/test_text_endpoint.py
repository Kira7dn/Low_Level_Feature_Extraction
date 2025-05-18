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
    from app.services.models import TextFeatures
    
    test_image_path = os.path.join(os.path.dirname(__file__), '..','test_images', 'text_sample.png')
    print(f"Using test image at: {test_image_path}")
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/text/extract",
            files={"file": (f.name, f, "image/png")}
        )
    
    # Status code validation
    if response.status_code != 200:
        try:
            print(f"Response content: {response.json()}")
        except Exception as e:
            print(f"Could not parse response as JSON: {response.content}")
            print(f"Error parsing JSON: {str(e)}")
        # Print request details for debugging
        print(f"Request URL: {response.request.url}")
        print(f"Request method: {response.request.method}")
        print(f"Request headers: {response.request.headers}")
        if hasattr(response.request, 'body') and response.request.body:
            print(f"Request body: {response.request.body}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Processing time validation
    processing_time = response.elapsed.total_seconds()
    assert validate_processing_time(processing_time), \
        f"Processing time {processing_time}s exceeds maximum limit"
    
    # Performance rating
    performance_rating = get_performance_rating(processing_time)
    print(f"Performance Rating: {performance_rating}")
    
    # Parse response
    result = response.json()
    print(f"Response content: {result}")
    
    # Convert TextFeatures to dict if needed
    if isinstance(result, TextFeatures):
        result = result.dict()
    
    # Check basic response structure
    assert "lines" in result, "Response must contain 'lines' field"
    assert isinstance(result["lines"], list), "'lines' must be a list"
    
    # Check we have at least one line of text
    assert len(result["lines"]) > 0, "Expected at least one line of text"
    
    # Check all text entries are non-empty strings
    for text in result["lines"]:
        assert isinstance(text, str), f"Expected text to be string, got {type(text)}"
        assert len(text.strip()) > 0, "Text entry cannot be empty"
    
    # Check if details are present (optional)
    if "details" in result:
        assert isinstance(result["details"], list), "'details' must be a list if present"
        for detail in result["details"]:
            assert isinstance(detail, dict), "Each detail must be a dictionary"
            assert "text" in detail, "Detail must contain 'text' field"
            if "confidence" in detail:
                assert isinstance(detail["confidence"], (int, float)), "Confidence must be a number"

def test_extract_text_invalid_format():
    """Test text extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..','test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/api/v1/text/extract",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
    assert "Unsupported image format" in str(data["detail"]["error"]["message"])

def test_extract_text_no_file():
    """Test text extraction with no file"""
    response = client.post("/api/v1/text/extract")
    
    assert response.status_code == 422  # Validation error for missing file
    data = response.json()
    assert isinstance(data['detail'], list)
    assert any(error.get('loc') == ['body', 'file'] for error in data['detail'])
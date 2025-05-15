import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.constants import (
    ValidationRules, 
    validate_hex_color, 
    validate_response_structure,
    validate_processing_time,
    get_performance_rating
)

# Test client setup
client = TestClient(app)

def test_extract_colors_success():
    """Validate successful color extraction"""
    test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'shapes_sample.png')
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/colors/extract",
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
        ValidationRules.COLOR_EXTRACTION["expected_keys"],
        value_types={
            "primary": str,
            "background": str,
            "accent": list
        },
        additional_checks=[
            lambda r: len(r["accent"]) <= ValidationRules.COLOR_EXTRACTION["max_accent_colors"]
        ]
    )
    assert is_valid, f"Invalid response structure: {error_msg}"
    
    # Color validation
    assert validate_hex_color(result["primary"]), "Primary color must be a valid hex color"
    assert validate_hex_color(result["background"]), "Background color must be a valid hex color"
    assert all(validate_hex_color(color) for color in result["accent"]), \
        "All accent colors must be valid hex colors"

def test_extract_colors_invalid_format():
    """Test color extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/api/v1/colors/extract",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "File format not supported" in str(data["detail"])

def test_extract_colors_no_file():
    """Test color extraction with no file"""
    response = client.post("/api/v1/colors/extract")
    
    assert response.status_code in [400, 422]  # Either validation error or missing file error
    data = response.json()
    assert "detail" in data

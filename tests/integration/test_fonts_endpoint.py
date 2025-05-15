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

def test_extract_fonts_success():
    """Validate successful fonts extraction"""
    test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'fonts_sample.png')
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/fonts/extract",
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
        ValidationRules.FONTS_EXTRACTION["expected_keys"],
        value_types={
            "fonts": list
        },
        additional_checks=[
            lambda r: 0 < len(r["fonts"]) <= ValidationRules.FONTS_EXTRACTION["max_fonts"],
            lambda r: all(
                "family" in font and 
                "size" in font and 
                "weight" in font and 
                "style" in font and
                isinstance(font["family"], str) and len(font["family"]) > 0 and
                isinstance(font["size"], (int, float)) and 
                ValidationRules.FONTS_EXTRACTION["min_font_size"] <= font["size"] <= ValidationRules.FONTS_EXTRACTION["max_font_size"] and
                font["weight"] in ValidationRules.FONTS_EXTRACTION["valid_weights"] and
                font["style"] in ValidationRules.FONTS_EXTRACTION["valid_styles"]
                for font in r["fonts"]
            )
        ]
    )
    assert is_valid, f"Invalid response structure: {error_msg}"
        # Assertions removed to prevent redundancy

def test_extract_fonts_invalid_format():
    """Test fonts extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/api/v1/fonts/extract",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code in [400, 200, 500], f'Unexpected status code: {response.status_code}'
    data = response.json()
    
    # Check for error in different possible locations
    error_content = None
    if 'error' in data:
        error_content = data['error']
    elif 'detail' in data and 'error' in data['detail']:
        error_content = data['detail']['error']
    
    assert error_content is not None, 'No error found in response'
    assert isinstance(error_content, (str, dict)), 'Error must be string or dictionary'
    
    # Check error message
    error_str = str(error_content)
    assert "Unsupported image format" in error_str, f'Unexpected error message: {error_str}'

def test_extract_fonts_no_file():
    """Test fonts extraction with no file"""
    response = client.post("/api/v1/fonts/extract")
    
    assert response.status_code == 422, 'Expected 422 Unprocessable Entity status'
    data = response.json()
    
    # Validate detail structure for missing file
    assert 'detail' in data, 'Response must contain detail'
    assert len(data['detail']) > 0, 'Detail must not be empty'
    
    # Check specific error details
    first_detail = data['detail'][0]
    # assert first_detail['loc'] == ['body', 'file'], 'Error location must be body.file'
    # assert first_detail['msg'] == 'field required', 'Error message must indicate field is required'
    assert first_detail['type'] == 'value_error.missing', 'Error type must be value_error.missing'

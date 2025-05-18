import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Test client setup
client = TestClient(app)
from tests.constants import (
    ValidationRules, 
    validate_response_structure,
    validate_processing_time,
    get_performance_rating
)

def test_extract_shapes_success():
    """Validate successful shapes extraction"""
    test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'shapes_sample.png')
    
    client = TestClient(app)
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/shapes/detect",
            files={"file": (f.name, f, "image/png")}
        )
        # Status code validation
        # print(f"Response status code: {response.status_code}")
        # print(f"Response headers: {response.headers}")
        # print(f"Response content: {response.json()}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Processing time validation
    processing_time = response.elapsed.total_seconds()
    assert validate_processing_time(processing_time), \
        f"Processing time {processing_time}s exceeds maximum limit"
    
    # Performance rating
    # performance_rating = get_performance_rating(processing_time)
    # print(f"Performance Rating: {performance_rating}")
    
    # Response structure validation
    result = response.json()
    # Validate response structure
    assert "shapes" in result, "Shapes key missing"
    assert isinstance(result["shapes"], list), "Shapes must be a list"
    assert 0 < len(result["shapes"]) <= ValidationRules.SHAPES_EXTRACTION["max_shapes"], "Invalid number of shapes"
    
    # Validate each shape
    for shape in result["shapes"]:
        assert "type" in shape, f"Shape missing type: {shape}"
        assert shape["type"] in ValidationRules.SHAPES_EXTRACTION["valid_shape_types"], f"Invalid shape type: {shape['type']}"
        
        # Ensure shape has coordinates or x/y/width/height
        assert any(key in shape for key in ["coordinates", "x", "y", "width", "height"]), f"Shape missing coordinate information: {shape}"
    
    # Validate total_shapes and metadata
    assert "total_shapes" in result, "Total shapes key missing"
    assert "metadata" in result, "Metadata key missing"
    assert isinstance(result["total_shapes"], int), "Total shapes must be an integer"
    assert isinstance(result["metadata"], dict), "Metadata must be a dictionary"

def test_extract_shapes_invalid_format():
    """Test shapes extraction with an invalid file format"""
    test_text_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'invalid.txt')
    
    with open(test_text_path, "rb") as f:
        response = client.post(
            "/api/v1/shapes/detect",
            files={"file": ("invalid.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]
    assert data["detail"]["error"]["code"] == "INVALID_IMAGE_TYPE"
    assert "Unsupported image format" in data["detail"]["error"]["message"]

def test_extract_shapes_no_file():
    """Test shapes extraction with no file"""
    response = client.post("/api/v1/shapes/detect")
    
    print(f"Response status code: {response.status_code}")
    # print(f"Response content: {response.json()}")
    
    assert response.status_code in [400, 422]  # Either validation error or missing file error
    data = response.json()
    assert isinstance(data, dict), "Expected a dictionary of errors"
    assert "detail" in data, "No 'detail' key in error response"
    assert isinstance(data["detail"], list), "Expected 'detail' to be a list of errors"
    assert len(data["detail"]) > 0, "No validation errors found"
    assert all('loc' in error and 'msg' in error for error in data["detail"]), "Invalid error structure"
    assert any(error['loc'] == ['body', 'file'] for error in data["detail"]), "No file-related validation error found"

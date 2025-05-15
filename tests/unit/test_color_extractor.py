import os
import numpy as np
import pytest
import time
from app.services.color_extractor import ColorExtractor
from app.services.image_processor import ImageProcessor
from tests.constants import (
    validate_response_structure,
    validate_processing_time,
    ValidationRules
)

def test_color_extractor_initialization():
    """Test ColorExtractor initialization"""
    extractor = ColorExtractor()
    assert extractor is not None

def test_extract_primary_color():
    """Test extracting primary color from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'sample_design.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract colors
    extractor = ColorExtractor()
    start_time = time.time()
    colors = extractor.extract_colors(cv_image)
    elapsed_time = time.time() - start_time

    # Validate response structure
    result = {"colors": colors}
    is_valid, error_msg = validate_response_structure(
        result,
        expected_keys=["colors"],
        value_types={"colors": dict},
        context="color_extraction"
    )
    assert is_valid, error_msg

    # Validate color dictionary structure
    color_info = result["colors"]
    assert "primary" in color_info, "Missing primary color"
    assert "background" in color_info, "Missing background color"
    assert "accent" in color_info, "Missing accent colors"
    
    # Validate color formats (hex codes)
    assert isinstance(color_info["primary"], str), "Primary color should be a string"
    assert isinstance(color_info["background"], str), "Background color should be a string"
    assert isinstance(color_info["accent"], list), "Accent colors should be a list"
    
    # Validate hex color format
    hex_pattern = ValidationRules.COLOR_EXTRACTION["hex_color_pattern"]
    assert hex_pattern.match(color_info["primary"]), "Invalid primary color format"
    assert hex_pattern.match(color_info["background"]), "Invalid background color format"
    assert all(hex_pattern.match(color) for color in color_info["accent"]), "Invalid accent color format"
    assert len(color_info["accent"]) <= ValidationRules.COLOR_EXTRACTION["max_accent_colors"], "Too many accent colors"

    # Validate processing time
    is_valid, error_msg = validate_processing_time(
        elapsed_time,
        context="color_extraction"
    )
    assert is_valid, error_msg

def test_color_extractor_empty_image():
    """Test color extraction on an empty or invalid image"""
    extractor = ColorExtractor()
    
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract colors
    start_time = time.time()
    colors = extractor.extract_colors(empty_image)
    elapsed_time = time.time() - start_time

    # Validate response structure
    result = {"colors": colors}
    is_valid, error_msg = validate_response_structure(
        result,
        expected_keys=["colors"],
        value_types={"colors": dict},
        context="color_extraction"
    )
    assert is_valid, error_msg

    # Validate color dictionary structure
    color_info = result["colors"]
    assert "primary" in color_info, "Missing primary color"
    assert "background" in color_info, "Missing background color"
    assert "accent" in color_info, "Missing accent colors"

    # Validate processing time
    is_valid, error_msg = validate_processing_time(
        elapsed_time,
        context="color_extraction"
    )
    assert is_valid, error_msg

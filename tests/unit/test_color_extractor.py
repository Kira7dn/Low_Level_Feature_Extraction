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
    color_palette = extractor.extract_colors(cv_image)
    elapsed_time = time.time() - start_time

    # Convert ColorPalette to dict for validation
    color_dict = color_palette.dict()
    
    # Validate response structure
    result = {"colors": color_dict}
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
    assert color_info["primary"] is None or isinstance(color_info["primary"], str), "Primary color should be a string or None"
    assert color_info["background"] is None or isinstance(color_info["background"], str), "Background color should be a string or None"
    assert isinstance(color_info["accent"], list), "Accent colors should be a list"
    
    # Validate hex color format for non-None values
    if color_info["primary"]:
        assert color_info["primary"].startswith('#'), f"Primary color {color_info['primary']} should start with #"
        assert len(color_info["primary"]) in [4, 7], f"Invalid hex color format: {color_info['primary']}"  # #RGB or #RRGGBB
    
    if color_info["background"]:
        assert color_info["background"].startswith('#'), f"Background color {color_info['background']} should start with #"
        assert len(color_info["background"]) in [4, 7], f"Invalid hex color format: {color_info['background']}"
    
    for color in color_info["accent"]:
        assert color.startswith('#'), f"Accent color {color} should start with #"
        assert len(color) in [4, 7], f"Invalid hex color format: {color}"
    
    # Validate processing time
    is_valid_time = validate_processing_time(elapsed_time)
    assert is_valid_time, "Color extraction took too long"
    
    # Additional validation for color extraction specific rules
    max_accent_colors = ValidationRules.COLOR_EXTRACTION["max_accent_colors"]
    assert len(color_info["accent"]) <= max_accent_colors, f"Too many accent colors (max {max_accent_colors})"
    
    # Validate color contrast if both primary and background are present
    if color_info["primary"] and color_info["background"]:
        # Simple check that primary and background are not too similar
        assert color_info["primary"] != color_info["background"], "Primary and background colors should be different"

def test_color_extractor_empty_image():
    """Test color extraction on an empty or invalid image"""
    extractor = ColorExtractor()
    
    # Test with None
    color_palette = extractor.extract_colors(None)
    assert color_palette is not None
    assert hasattr(color_palette, 'primary')
    assert hasattr(color_palette, 'background')
    assert hasattr(color_palette, 'accent')
    
    # Test with empty array
    empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
    color_palette = extractor.extract_colors(empty_image)
    assert color_palette is not None
    assert hasattr(color_palette, 'primary')
    assert hasattr(color_palette, 'background')
    assert hasattr(color_palette, 'accent')
    
    # Test with invalid image data
    invalid_image = np.array([[[300, 400, 500]]], dtype=np.uint16)  # Out of range values
    color_palette = extractor.extract_colors(invalid_image)
    assert color_palette is not None
    assert hasattr(color_palette, 'primary')
    assert hasattr(color_palette, 'background')
    assert hasattr(color_palette, 'accent')
    
    # Test with malformed array
    malformed_image = np.zeros((10, 10, 5), dtype=np.uint8)  # 5 channels
    color_palette = extractor.extract_colors(malformed_image)
    assert color_palette is not None
    assert hasattr(color_palette, 'primary')
    assert hasattr(color_palette, 'background')
    assert hasattr(color_palette, 'accent')

    # Validate processing time
    start_time = time.time()
    extractor.extract_colors(np.zeros((100, 100, 3), dtype=np.uint8))
    elapsed_time = time.time() - start_time
    is_valid, error_msg = validate_processing_time(
        elapsed_time,
        context="color_extraction"
    )
    assert is_valid, error_msg

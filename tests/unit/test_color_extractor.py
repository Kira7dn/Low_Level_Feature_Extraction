import os
import numpy as np
import pytest
from app.services.color_extractor import ColorExtractor
from app.services.image_processor import ImageProcessor

def test_color_extractor_initialization():
    """Test ColorExtractor initialization"""
    extractor = ColorExtractor()
    assert extractor is not None

def test_extract_primary_color():
    """Test extracting primary color from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'sample_design.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract colors
    extractor = ColorExtractor()
    result = extractor.extract_colors(cv_image)
    
    # Validate result structure
    assert "primary" in result
    assert "background" in result
    assert "accent" in result
    
    # Validate color formats (hex codes)
    assert isinstance(result["primary"], str)
    assert isinstance(result["background"], str)
    assert isinstance(result["accent"], list)
    
    # Validate hex color format
    def is_valid_hex(color):
        return len(color) == 7 and color.startswith('#')
    
    assert is_valid_hex(result["primary"])
    assert is_valid_hex(result["background"])
    assert all(is_valid_hex(color) for color in result["accent"])

def test_color_extractor_empty_image():
    """Test color extraction on an empty or invalid image"""
    extractor = ColorExtractor()
    
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract colors
    result = extractor.extract_colors(empty_image)
    
    # Validate fallback or default behavior
    assert "primary" in result
    assert "background" in result
    assert "accent" in result

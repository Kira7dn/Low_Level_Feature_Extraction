import os
import numpy as np
import pytest
from app.services.font_detector import FontDetector
from app.services.image_processor import ImageProcessor

def test_font_detector_initialization():
    """Test FontDetector initialization"""
    detector = FontDetector()
    assert detector is not None

def test_extract_fonts_from_sample_image():
    """Test extracting fonts from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'fonts_sample.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract fonts
    detector = FontDetector()
    result = detector.detect_font(cv_image)
    
    # Validate result structure
    assert isinstance(result, dict)
    
    # Validate font entries
    assert "font_family" in result
    assert "font_size" in result
    assert "font_weight" in result
    
    # Validate specific attributes
    assert isinstance(result["font_family"], str)
    assert isinstance(result["font_size"], int)
    assert result["font_weight"] in ["Light", "Regular", "Bold", "Unknown"]

def test_font_detector_empty_image():
    """Test font detection on an empty or invalid image"""
    detector = FontDetector()
    
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract fonts
    result = detector.detect_font(empty_image)
    
    # Validate fallback or default behavior
    assert isinstance(result, dict)
    assert result["font_family"] == "Unknown"
    assert result["font_size"] == 0
    assert result["font_weight"] == "Unknown"

def test_font_detector_low_resolution_image():
    """Test font detection on a low resolution image"""
    detector = FontDetector()
    
    # Create a low resolution image with text-like patterns
    low_res_image = np.zeros((50, 50, 3), dtype=np.uint8)
    low_res_image[:25, :25] = 255  # White background
    low_res_image[10:20, 10:20] = 0  # Black text-like region
    
    # Extract fonts
    result = detector.detect_font(low_res_image)
    
    # Validate result
    assert isinstance(result, dict)
    # Validate dictionary keys exist
    assert "font_family" in result
    assert "font_size" in result
    assert "font_weight" in result
    # Might find fonts or might not, but should not raise an exception

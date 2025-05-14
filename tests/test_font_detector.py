import cv2
import numpy as np
import pytest
from app.services.font_detector import FontDetector

def create_synthetic_text_image(
    text_color=(0, 0, 0),  # Black text
    bg_color=(255, 255, 255),  # White background
    size=(300, 300),
    text_region_size=(100, 50)
):
    """
    Create a synthetic image with a text-like region
    
    Args:
        text_color (tuple): Color of text region
        bg_color (tuple): Background color
        size (tuple): Total image size
        text_region_size (tuple): Size of text region
    
    Returns:
        np.ndarray: Synthetic image
    """
    # Create blank image
    img = np.full((size[0], size[1], 3), bg_color, dtype=np.uint8)
    
    # Create text-like region
    x = (size[0] - text_region_size[0]) // 2
    y = (size[1] - text_region_size[1]) // 2
    cv2.rectangle(
        img, 
        (x, y), 
        (x + text_region_size[0], y + text_region_size[1]), 
        text_color, 
        -1
    )
    
    return img

def test_preprocess_image():
    """Test image preprocessing method"""
    img = create_synthetic_text_image()
    binary = FontDetector.preprocess_image(img)
    
    # Check output is binary image
    assert binary.dtype == np.uint8
    assert len(binary.shape) == 2  # Grayscale
    assert binary.min() == 0
    assert binary.max() == 255

def test_detect_text_regions():
    """Test text region detection"""
    # Create image with clear text-like region
    img = create_synthetic_text_image()
    binary = FontDetector.preprocess_image(img)
    
    # Detect text regions
    regions = FontDetector.detect_text_regions(binary)
    
    # Verify regions are detected
    assert len(regions) > 0
    
    # Check region format (x, y, width, height)
    for x, y, w, h in regions:
        assert w > 0
        assert h > 0
        assert x >= 0
        assert y >= 0

def test_estimate_font_size():
    """Test font size estimation"""
    # Test various region heights
    test_heights = [10, 20, 50, 100]
    
    for height in test_heights:
        estimated_size = FontDetector.estimate_font_size(height)
        
        # Estimated size should be roughly 75% of region height
        assert 0 < estimated_size <= height
        assert estimated_size == int(height * 0.75)

def test_estimate_font_weight():
    """Test font weight estimation"""
    # Light region (near white)
    light_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    weight = FontDetector.estimate_font_weight(light_img)
    assert weight == "Light", f"Unexpected weight: {weight}"
    
    # Regular region (light gray)
    regular_img = np.full((100, 100, 3), 210, dtype=np.uint8)
    weight = FontDetector.estimate_font_weight(regular_img)
    assert weight == "Regular", f"Unexpected weight: {weight}"
    
    # Bold region (dark gray)
    bold_img = np.full((100, 100, 3), 180, dtype=np.uint8)
    weight = FontDetector.estimate_font_weight(bold_img)
    assert weight == "Bold", f"Unexpected weight: {weight}"

def test_detect_font_with_text_region():
    """Test full font detection with a text-like region"""
    # Create a synthetic image with a text-like region
    img = create_synthetic_text_image(
        text_color=(50, 50, 50),  # Dark gray
        bg_color=(255, 255, 255)
    )
    
    # Detect font
    font_info = FontDetector.detect_font(img)
    
    # Verify returned dictionary structure
    assert isinstance(font_info, dict)
    
    # Check font family
    assert "font_family" in font_info, "Font family should be present in results"
    assert isinstance(font_info["font_family"], str), "Font family should be a string"
    
    # Check font size
    assert "font_size" in font_info, "Font size should be present in results"
    assert isinstance(font_info["font_size"], int), "Font size should be an integer"
    assert font_info["font_size"] > 0, "Font size should be a positive number"
    
    # Check font weight
    assert "font_weight" in font_info, "Font weight should be present in results"
    assert font_info["font_weight"] in ["Light", "Regular", "Bold"], "Invalid font weight"
    
    # Specific test for dark gray text
    assert font_info["font_weight"] == "Bold", "Dark gray text should be classified as Bold"
    
    # Basic validation of values
    assert isinstance(font_info["font_family"], str)
    assert isinstance(font_info["font_size"], int)
    assert isinstance(font_info["font_weight"], str)
    assert font_info["font_size"] > 0

def test_detect_font_no_text():
    """Test font detection with an empty/blank image"""
    # Create a completely white image
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    
    # Detect font
    font_info = FontDetector.detect_font(img)
    
    # Verify default values for no text
    assert font_info["font_family"] == "Unknown"
    assert font_info["font_size"] == 0
    assert font_info["font_weight"] == "Unknown"

def test_identify_font_family():
    """Test font family identification"""
    # Create a synthetic text region
    img = create_synthetic_text_image()
    
    # Current implementation always returns "Arial"
    font_family = FontDetector.identify_font_family(img)
    assert font_family == "Arial"

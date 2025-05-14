import os
import numpy as np
import pytest
from app.services.shadow_analyzer import ShadowAnalyzer
from app.services.image_processor import ImageProcessor

def test_shadow_analyzer_initialization():
    """Test ShadowAnalyzer initialization"""
    analyzer = ShadowAnalyzer()
    assert analyzer is not None

def test_extract_shadow_level_from_sample_image():
    """Test extracting shadow level from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'shadows_sample.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract shadow level
    analyzer = ShadowAnalyzer()
    result = analyzer.analyze_shadow_level(cv_image)
    
    # Validate result
    assert isinstance(result, str)
    assert result in ["Low", "Moderate", "High"]

def test_shadow_analyzer_empty_image():
    """Test shadow analysis on an empty or invalid image"""
    analyzer = ShadowAnalyzer()
    
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract shadow level
    result = analyzer.analyze_shadow_level(empty_image)
    
    # Validate fallback or default behavior
    assert isinstance(result, str)
    assert result == "Low"  # Default for images with no discernible shadows

def test_shadow_analyzer_high_contrast_image():
    """Test shadow analysis on a high contrast image"""
    analyzer = ShadowAnalyzer()
    
    # Create a high contrast image
    high_contrast_image = np.zeros((100, 100, 3), dtype=np.uint8)
    high_contrast_image[:50, :50] = 0  # Black top-left
    high_contrast_image[50:, 50:] = 255  # White bottom-right
    
    # Extract shadow level
    result = analyzer.analyze_shadow_level(high_contrast_image)
    
    # Validate result
    assert isinstance(result, str)
    assert result in ["Low", "Moderate", "High"]

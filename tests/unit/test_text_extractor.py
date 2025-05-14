import os
import numpy as np
import pytest
from app.services.text_extractor import TextExtractor
from app.services.image_processor import ImageProcessor

def test_text_extractor_initialization():
    """Test TextExtractor initialization"""
    extractor = TextExtractor()
    assert extractor is not None

def test_extract_text_from_sample_image():
    """Test extracting text from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'text_sample.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract text
    extractor = TextExtractor()
    result = TextExtractor.extract_text(cv_image)
    
    # Validate result structure
    assert isinstance(result, dict)
    assert 'lines' in result
    assert isinstance(result['lines'], list)
    assert len(result['lines']) > 0
    
    # Validate text entries
    for text_entry in result['lines']:
        assert isinstance(text_entry, str)
        assert len(text_entry.strip()) > 0

def test_text_extractor_empty_image():
    """Test text extraction on an empty or invalid image"""
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract text
    result = TextExtractor.extract_text(empty_image)
    
    # Validate fallback or default behavior
    assert isinstance(result, dict)
    assert 'lines' in result
    assert isinstance(result['lines'], list)
    assert len(result['lines']) == 0  # No text found in empty image

def test_text_extractor_low_contrast_image():
    """Test text extraction on a low contrast image"""
    extractor = TextExtractor()
    
    # Create a low contrast image (grayscale gradient)
    low_contrast_image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        low_contrast_image[:, i] = i
    
    # Extract text
    result = TextExtractor.extract_text(low_contrast_image)
    
    # Validate result
    assert isinstance(result, dict)
    assert 'lines' in result
    assert isinstance(result['lines'], list)
    # Might find text or might not, but should not raise an exception

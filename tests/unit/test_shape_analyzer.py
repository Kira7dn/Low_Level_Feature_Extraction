import os
import numpy as np
import pytest
from app.services.shape_analyzer import ShapeAnalyzer
from app.services.image_processor import ImageProcessor

def test_shape_analyzer_initialization():
    """Test ShapeAnalyzer initialization"""
    analyzer = ShapeAnalyzer()
    assert analyzer is not None

def test_extract_shapes_from_sample_image():
    """Test extracting shapes from a sample image"""
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images', 'shapes_sample.png')
    
    # Load image
    image_bytes = open(test_image_path, 'rb').read()
    cv_image = ImageProcessor.load_cv2_image(image_bytes)
    
    # Extract shapes
    analyzer = ShapeAnalyzer()
    result = analyzer.extract_shapes(cv_image)
    
    # Validate result structure
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Validate shape entries
    for shape in result:
        assert "type" in shape
        assert "coordinates" in shape
        assert shape["type"] in ["rectangle", "circle", "polygon"]
        assert isinstance(shape["coordinates"], list)
        assert len(shape["coordinates"]) > 0

def test_shape_analyzer_empty_image():
    """Test shape analysis on an empty or invalid image"""
    analyzer = ShapeAnalyzer()
    
    # Create an empty numpy array
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Extract shapes
    result = analyzer.extract_shapes(empty_image)
    
    # Validate fallback or default behavior
    assert isinstance(result, list)
    assert len(result) == 0  # No shapes found in empty image

def test_shape_analyzer_noisy_image():
    """Test shape analysis on a noisy image"""
    analyzer = ShapeAnalyzer()
    
    # Create a noisy image
    noisy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Extract shapes
    result = analyzer.extract_shapes(noisy_image)
    
    # Validate result
    assert isinstance(result, list)
    # Might find shapes or might not, but should not raise an exception

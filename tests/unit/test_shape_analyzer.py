import os
import cv2
import time
import numpy as np
import pytest
from app.services.shape_analyzer import ShapeAnalyzer
from app.services.image_processor import ImageProcessor
from tests.constants import validate_response_structure, validate_processing_time

class TestShapeAnalyzer:
    @pytest.fixture
    def sample_shapes_image(self):
        """
        Create a sample image with various shapes for testing
        """
        # Create a blank white image
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        image.fill(255)
        
        # Draw a rectangle
        cv2.rectangle(image, (50, 50), (200, 150), (0, 0, 0), 2)
        
        # Draw a square
        cv2.rectangle(image, (250, 50), (350, 150), (0, 0, 0), 2)
        
        # Draw a triangle
        pts = np.array([(450, 50), (500, 150), (550, 50)], np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 0), 2)
        
        # Draw a circle
        cv2.circle(image, (150, 250), 50, (0, 0, 0), 2)
        
        return image

    def test_shape_analyzer_initialization(self):
        """Test ShapeAnalyzer initialization"""
        analyzer = ShapeAnalyzer()
        assert analyzer is not None

    def test_extract_shapes_from_sample_image(self):
        """Test extracting shapes from a sample image"""
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'shapes_sample.png')
        
        # Load image
        image_bytes = open(test_image_path, 'rb').read()
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        
        # Extract shapes
        analyzer = ShapeAnalyzer()
        start_time = time.time()
        result = analyzer.extract_shapes(cv_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            {"shapes": result},
            expected_keys=["shapes"],
            value_types={
                "shapes": list
            },
            context="shape_extraction"
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="shape_extraction"
        )
        assert is_valid, error_msg
        assert len(result) > 0
        
        # Validate shape entries
        for shape in result:
            assert "type" in shape
            assert "coordinates" in shape
            assert shape["type"] in ["rectangle", "circle", "polygon"]
            assert isinstance(shape["coordinates"], list)
            assert len(shape["coordinates"]) > 0

    def test_preprocess_image(self, sample_shapes_image):
        """
        Test image preprocessing for shape detection
        """
        preprocessed = ShapeAnalyzer.preprocess_image(sample_shapes_image)
        
        # Check preprocessing results
        assert preprocessed is not None
        assert preprocessed.shape == sample_shapes_image.shape[:2]
        assert preprocessed.dtype == np.uint8
        assert len(preprocessed.shape) == 2  # 2D binary image

    def test_detect_border_radius(self, sample_shapes_image):
        """
        Test border radius detection
        """
        # Preprocess the image
        preprocessed = ShapeAnalyzer.preprocess_image(sample_shapes_image)
        
        # Find contours
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Test border radius detection for each contour
        for contour in contours:
            radius = ShapeAnalyzer.detect_border_radius(contour)
            assert isinstance(radius, float)
            assert radius >= 0

    def test_edge_cases(self):
        """
        Test shape analysis with various edge cases
        """
        # Test with empty image
        empty_image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = ShapeAnalyzer.analyze_shapes(empty_image)
        assert result["total_shapes"] == 0
        
        # Test with very noisy image
        noisy_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        result = ShapeAnalyzer.analyze_shapes(noisy_image)
        # Shapes detected might vary, but shouldn't crash
        assert isinstance(result, dict)
        assert "shapes" in result
        assert "total_shapes" in result

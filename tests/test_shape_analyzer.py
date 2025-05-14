import pytest
import cv2
import numpy as np
from app.services.shape_analyzer import ShapeAnalyzer

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
        
        # Flag to ensure at least one contour has a meaningful border radius
        found_meaningful_radius = False
        
        # Test border radius for each contour
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            border_radius = ShapeAnalyzer.detect_border_radius(contour)
            
            # Border radius should be a non-negative float
            assert isinstance(border_radius, float)
            assert border_radius >= 0
            
            # Check if we found a meaningful border radius
            if border_radius > 0:
                found_meaningful_radius = True
        
        # Ensure we tested at least one contour with a potentially rounded shape
        assert found_meaningful_radius, "No contours with meaningful border radius found"

    def test_analyze_shapes(self, sample_shapes_image):
        """
        Test comprehensive shape analysis
        """
        # Analyze shapes
        result = ShapeAnalyzer.analyze_shapes(sample_shapes_image)
        
        # Validate result structure
        assert isinstance(result, dict)
        assert "shapes" in result
        assert "total_shapes" in result
        assert "metadata" in result
        
        # Check metadata
        assert result["metadata"]["image_width"] == 600
        assert result["metadata"]["image_height"] == 400
        
        # Validate shape detection
        shapes = result["shapes"]
        assert result["total_shapes"] > 0
        
        # Check shape types
        shape_types = [shape["type"] for shape in shapes]
        expected_types = {"rectangle", "square", "triangle", "circle"}
        assert any(shape_type in expected_types for shape_type in shape_types)
        
        # Validate shape properties
        for shape in shapes:
            assert "x" in shape
            assert "y" in shape
            assert "width" in shape
            assert "height" in shape
            assert "border_radius" in shape
            assert "area" in shape
            assert shape["border_radius"] >= 0
            assert shape["area"] > 0

    def test_performance(self, sample_shapes_image):
        """
        Test performance of shape analysis
        """
        import time
        
        # Measure processing time
        start_time = time.time()
        result = ShapeAnalyzer.analyze_shapes(sample_shapes_image)
        processing_time = time.time() - start_time
        
        # Ensure processing is quick
        assert processing_time < 1.0, f"Processing took {processing_time} seconds, which is too slow"
        
        # Ensure meaningful results
        assert result["total_shapes"] > 0

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

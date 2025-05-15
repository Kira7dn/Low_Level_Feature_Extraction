import os
import cv2
import time
import numpy as np
import pytest
from app.services.shadow_analyzer import ShadowAnalyzer
from app.services.image_processor import ImageProcessor
from tests.constants import validate_response_structure, validate_processing_time

class TestShadowAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ShadowAnalyzer()
    
    @pytest.fixture
    def create_test_image(self):
        def _create(intensity=50, size=(100, 100)):
            img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
            # Add a rectangle shadow region
            cv2.rectangle(img, (30, 30), (70, 70), (intensity, intensity, intensity), -1)
            return img
        return _create

    @pytest.fixture
    def white_image(self):
        return np.full((100, 100, 3), 255, dtype=np.uint8)

    def test_initialization(self, analyzer):
        """Test ShadowAnalyzer initialization"""
        assert analyzer is not None

    def test_real_image(self, analyzer):
        """Test extracting shadow level from a sample image"""
        test_image_path = os.path.join(os.path.dirname(__file__),'..', 'test_images', 'shadows_sample.png')
        
        # Load image
        image_bytes = open(test_image_path, 'rb').read()
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        
        # Extract and validate shadow level
        start_time = time.time()
        shadow_level = analyzer.analyze_shadow_level(cv_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"shadow_level": shadow_level}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["shadow_level"],
            value_types={"shadow_level": str},
            context="shadow_analysis"
        )
        assert is_valid, error_msg
        assert result["shadow_level"] in ["Low", "Moderate", "High"]
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="shadow_analysis"
        )
        assert is_valid, error_msg

    def test_no_shadow(self, analyzer, white_image):
        """Test image with no shadows"""
        start_time = time.time()
        shadow_level = analyzer.analyze_shadow_level(white_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"shadow_level": shadow_level}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["shadow_level"],
            value_types={"shadow_level": str},
            context="shadow_analysis"
        )
        assert is_valid, error_msg
        assert result["shadow_level"] == "Low"
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="shadow_analysis"
        )
        assert is_valid, error_msg

    def test_high_shadow(self, analyzer, create_test_image):
        """Test image with high shadow intensity"""
        img = create_test_image(intensity=30)
        start_time = time.time()
        shadow_level = analyzer.analyze_shadow_level(img)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"shadow_level": shadow_level}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["shadow_level"],
            value_types={"shadow_level": str},
            context="shadow_analysis"
        )
        assert is_valid, error_msg
        assert result["shadow_level"] == "High"
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="shadow_analysis"
        )
        assert is_valid, error_msg

    def test_moderate_shadow(self, analyzer, create_test_image):
        """Test image with moderate shadow intensity"""
        img = create_test_image(intensity=210)
        result = analyzer.analyze_shadow_level(img)
        assert result == 'Moderate'

    def test_low_shadow(self, analyzer, create_test_image):
        """Test image with low shadow intensity"""
        img = create_test_image(intensity=240)
        result = analyzer.analyze_shadow_level(img)
        assert result == 'Low'

    def test_empty_image(self, analyzer):
        """Test shadow analysis on an empty image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = analyzer.analyze_shadow_level(empty_image)
        assert result == "Low"  # Default for images with no discernible shadows

    def test_high_contrast_image(self, analyzer):
        """Test shadow analysis on a high contrast image"""
        high_contrast_image = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast_image[:50, :50] = 0  # Black top-left
        high_contrast_image[50:, 50:] = 255  # White bottom-right
        
        result = analyzer.analyze_shadow_level(high_contrast_image)
        assert isinstance(result, str)
        assert result in ["Low", "Moderate", "High"]

import os
import cv2
import numpy as np
import pytest
import time
from typing import Dict, Tuple
from app.services.font_detector import FontDetector
from app.services.image_processor import ImageProcessor
from tests.constants import validate_response_structure, validate_processing_time

@pytest.fixture(scope="session")
def font_weights() -> Dict[str, int]:
    """Common font weight values for testing"""
    return {
        'light': 255,  # Near white
        'regular': 210,  # Light gray
        'bold': 180,    # Dark gray
    }

@pytest.fixture(scope="session")
def image_sizes() -> Dict[str, Tuple[int, int]]:
    """Common image sizes for testing"""
    return {
        'standard': (300, 300),
        'low_res': (50, 50),
        'text_region': (100, 50)
    }

class TestFontDetector:
    @pytest.fixture(scope="class")
    def detector(self) -> FontDetector:
        """Shared FontDetector instance for all tests"""
        return FontDetector()
    
    @pytest.fixture
    def create_test_image(self, image_sizes):
        def _create(
            text_color: Tuple[int, int, int] = (0, 0, 0),  # Black text
            bg_color: Tuple[int, int, int] = (255, 255, 255),  # White background
            size: Tuple[int, int] = image_sizes['standard'],
            text_region_size: Tuple[int, int] = image_sizes['text_region']
        ) -> np.ndarray:
            """Create a synthetic test image with a text region
            
            Args:
                text_color: RGB color for text region
                bg_color: RGB color for background
                size: Image dimensions (width, height)
                text_region_size: Text region dimensions (width, height)
            
            Returns:
                np.ndarray: Synthetic test image
            """
            image = np.full((*size, 3), bg_color, dtype=np.uint8)
            x = (size[0] - text_region_size[0]) // 2
            y = (size[1] - text_region_size[1]) // 2
            image[y:y+text_region_size[1], x:x+text_region_size[0]] = text_color
            return image
        return _create

    @pytest.fixture
    def sample_image(self, create_test_image):
        return create_test_image()

    @pytest.fixture
    def preprocessed_image(self, sample_image, detector):
        return FontDetector.preprocess_image(sample_image)
    
    def test_preprocess_image(self, preprocessed_image):
        """Test image preprocessing method produces valid grayscale output"""
        # Validate response structure
        result = {"image": preprocessed_image}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": np.ndarray},
            context="font_detection"
        )
        assert is_valid, error_msg

        # Verify image properties
        assert preprocessed_image.dtype == np.uint8, f"Expected uint8 dtype, got {preprocessed_image.dtype}"
        assert len(preprocessed_image.shape) == 2, f"Expected grayscale (2D), got shape {preprocessed_image.shape}"
        assert preprocessed_image.min() == 0, f"Expected min value 0, got {preprocessed_image.min()}"
        assert preprocessed_image.max() == 255, f"Expected max value 255, got {preprocessed_image.max()}"

    def test_detect_text_regions(self, preprocessed_image):
        """Test text region detection and validation"""
        start_time = time.time()
        regions = FontDetector.detect_text_regions(preprocessed_image)
        elapsed_time = time.time() - start_time

        # Validate response structure
        result = {"regions": regions}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["regions"],
            value_types={"regions": list},
            context="font_detection"
        )
        assert is_valid, error_msg

        # Verify regions are detected
        assert len(regions) > 0, "No text regions detected"
    
        # Validate each region's dimensions
        for i, (x, y, w, h) in enumerate(regions):
            assert w > 0, f"Region {i}: Invalid width {w}"
            assert h > 0, f"Region {i}: Invalid height {h}"
            assert x >= 0, f"Region {i}: Invalid x-coordinate {x}"
            assert y >= 0, f"Region {i}: Invalid y-coordinate {y}"
            # Verify region is within image bounds
            assert x + w <= preprocessed_image.shape[1], f"Region {i} extends beyond image width"
            assert y + h <= preprocessed_image.shape[0], f"Region {i} extends beyond image height"

        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="font_detection"
        )
        assert is_valid, error_msg

    @pytest.mark.parametrize('height', [10, 20, 50, 100])
    def test_estimate_font_size(self, detector, height):
        """Test font size estimation for various heights"""
        estimated_size = FontDetector.estimate_font_size(height)
        expected_size = int(height * 0.75)
        
        assert 0 < estimated_size <= height, f"Size {estimated_size} should be between 0 and {height}"
        assert estimated_size == expected_size, f"Expected size {expected_size}, got {estimated_size}"

    @pytest.fixture
    def test_images(self, font_weights) -> Dict[str, np.ndarray]:
        """Generate test images for different font weights"""
        return {
            weight: np.full((100, 100, 3), value, dtype=np.uint8)
            for weight, value in font_weights.items()
        }

    @pytest.mark.parametrize('weight_type,expected', [
        ('light', 'Light'),
        ('regular', 'Regular'),
        ('bold', 'Bold')
    ])
    def test_estimate_font_weight(self, detector, test_images, weight_type, expected):
        """Test font weight estimation for different weights"""
        weight = FontDetector.estimate_font_weight(test_images[weight_type])
        assert weight == expected, f"Expected {expected} for {weight_type}, got {weight}"

    @pytest.fixture
    def dark_text_image(self, create_test_image):
        return create_test_image(
            text_color=(50, 50, 50),  # Dark gray
            bg_color=(255, 255, 255)
        )

    @pytest.fixture
    def expected_font_info_keys(self) -> Dict[str, type]:
        """Expected keys and their types in font detection results"""
        return {
            'font_family': str,
            'font_size': int,
            'font_weight': str
        }

    def test_detect_font_with_text_region(self, detector, dark_text_image, expected_font_info_keys):
        """Test full font detection with a text-like region"""
        start_time = time.time()
        font_info = FontDetector.detect_font(dark_text_image)
        elapsed_time = time.time() - start_time

        # Validate response structure
        result = {"font_info": font_info}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["font_info"],
            value_types={"font_info": dict},
            context="font_detection"
        )
        assert is_valid, error_msg
        
        # Verify dictionary structure and types
        for key, expected_type in expected_font_info_keys.items():
            assert key in font_info, f"Missing required key: {key}"
            assert isinstance(font_info[key], expected_type), \
                f"Key {key}: expected type {expected_type}, got {type(font_info[key])}"
        
        # Validate font size
        assert font_info["font_size"] > 0, f"Invalid font size: {font_info['font_size']}"
        
        # Validate font weight
        valid_weights = ["Light", "Regular", "Bold"]

        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="font_detection"
        )
        assert is_valid, error_msg
        assert font_info["font_weight"] in valid_weights, \
            f"Invalid weight '{font_info['font_weight']}', expected one of {valid_weights}"
        
        # Verify dark gray text is detected as Bold
        assert font_info["font_weight"] == "Bold", \
            f"Dark gray text should be Bold, got {font_info['font_weight']}"

    @pytest.fixture
    def blank_image(self):
        return np.full((300, 300, 3), 255, dtype=np.uint8)

    @pytest.mark.parametrize('key,expected_value', [
        ('font_family', 'Unknown'),
        ('font_size', 0),
        ('font_weight', 'Unknown')
    ])
    def test_detect_font_no_text(self, detector, blank_image, key, expected_value):
        """Test font detection defaults for empty/blank image"""
        font_info = FontDetector.detect_font(blank_image)
        
        assert font_info[key] == expected_value, \
            f"Expected {key}='{expected_value}', got '{font_info[key]}'"

    def test_identify_font_family(self, detector, sample_image):
        """Test font family identification"""
        # Current implementation always returns "Arial"
        font_family = FontDetector.identify_font_family(sample_image)
        assert font_family == "Arial"

    @pytest.fixture
    def sample_image_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'test_images', 'fonts_sample.png')

    def test_extract_fonts_from_sample_image(self, detector, sample_image_path, expected_font_info_keys):
        """Test font detection on a real sample image"""
        try:
            with open(sample_image_path, 'rb') as f:
                image_bytes = f.read()
        except IOError as e:
            pytest.fail(f"Failed to read sample image: {e}")
            
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        result = detector.detect_font(cv_image)
        
        # Validate structure and types
        for key, expected_type in expected_font_info_keys.items():
            assert key in result, f"Missing required key: {key}"
            assert isinstance(result[key], expected_type), \
                f"Key {key}: expected type {expected_type}, got {type(result[key])}"
        
        # Validate font weight
        valid_weights = ["Light", "Regular", "Bold", "Unknown"]
        assert result["font_weight"] in valid_weights, \
            f"Invalid weight '{result['font_weight']}', expected one of {valid_weights}"

    @pytest.fixture
    def low_res_image(self, create_test_image, image_sizes) -> np.ndarray:
        """Create a low resolution test image"""
        return create_test_image(
            size=image_sizes['low_res'],
            text_region_size=(10, 10),
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255)
        )

    def test_font_detector_low_resolution_image(self, detector, low_res_image, expected_font_info_keys):
        """Test font detection handles low resolution images gracefully"""
        result = detector.detect_font(low_res_image)
        
        # Validate structure and types
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        
        for key, expected_type in expected_font_info_keys.items():
            assert key in result, f"Missing required key: {key}"
            assert isinstance(result[key], expected_type), \
                f"Key {key}: expected type {expected_type}, got {type(result[key])}"
        
        # Additional validation for low-res specific behavior could be added here
        # For example, checking if font size is proportional to image size

import pytest
import io
import time
from PIL import Image, ImageFilter
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
from tests.constants import validate_response_structure, validate_processing_time
from app.services.image_transformer import ImageTransformer

class TestImageTransformer:
    @pytest.fixture
    def image_size(self) -> Tuple[int, int]:
        """Standard test image size"""
        return (200, 200)
    
    @pytest.fixture
    def sample_pil_image(self, image_size: Tuple[int, int]) -> Image.Image:
        """Create a sample PIL image in RGB format"""
        return Image.new('RGB', image_size, color='red')
    
    @pytest.fixture
    def sample_cv2_image(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Create a sample OpenCV image in BGR format"""
        img = np.zeros((*image_size, 3), dtype=np.uint8)
        img[:] = [0, 0, 255]  # Blue in BGR
        return img
    
    @pytest.mark.parametrize('target_size,maintain_aspect', [
        ((100, None), True),    # Width only with aspect ratio
        ((None, 100), True),    # Height only with aspect ratio
        ((50, 50), False),      # Both dimensions without aspect ratio
        ((150, 75), True),      # Both dimensions with aspect ratio
    ])
    def test_resize_pil_image(self, sample_pil_image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool):
        """Test PIL image resizing with various configurations"""
        start_time = time.time()
        resized = ImageTransformer.resize(
            sample_pil_image, 
            width=target_size[0], 
            height=target_size[1], 
            maintain_aspect_ratio=maintain_aspect
        )
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": resized}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": Image.Image},
            context="image_transformation"
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_transformation"
        )
        assert is_valid, error_msg
        
        # Verify dimensions
        if target_size[0]:
            if maintain_aspect:
                assert resized.width <= target_size[0], f"Width {resized.width} exceeds target {target_size[0]}"
            else:
                assert resized.width == target_size[0], f"Width {resized.width} should equal target {target_size[0]}"
                
        if target_size[1]:
            if maintain_aspect:
                assert resized.height <= target_size[1], f"Height {resized.height} exceeds target {target_size[1]}"
            else:
                assert resized.height == target_size[1], f"Height {resized.height} should equal target {target_size[1]}"
        
        # Check aspect ratio preservation
        if maintain_aspect:
            original_ratio = sample_pil_image.width / sample_pil_image.height
            resized_ratio = resized.width / resized.height
            assert abs(original_ratio - resized_ratio) < 0.1, "Aspect ratio should be preserved"

    @pytest.mark.parametrize('target_size,maintain_aspect', [
        ((100, None), True),    # Width only with aspect ratio
        ((None, 100), True),    # Height only with aspect ratio
        ((50, 50), False),      # Both dimensions without aspect ratio
        ((150, 75), True),      # Both dimensions with aspect ratio
    ])
    def test_resize_cv2_image(self, sample_cv2_image: np.ndarray, target_size: Tuple[int, int], maintain_aspect: bool):
        """Test OpenCV image resizing with various configurations"""
        start_time = time.time()
        resized = ImageTransformer.resize(
            sample_cv2_image, 
            width=target_size[0], 
            height=target_size[1], 
            maintain_aspect_ratio=maintain_aspect
        )
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": resized}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": np.ndarray},
            context="image_transformation"
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_transformation"
        )
        assert is_valid, error_msg
        
        # Verify dimensions
        if target_size[0]:
            if maintain_aspect:
                assert resized.shape[1] <= target_size[0], f"Width {resized.shape[1]} exceeds target {target_size[0]}"
            else:
                assert resized.shape[1] == target_size[0], f"Width {resized.shape[1]} should equal target {target_size[0]}"
                
        if target_size[1]:
            if maintain_aspect:
                assert resized.shape[0] <= target_size[1], f"Height {resized.shape[0]} exceeds target {target_size[1]}"
            else:
                assert resized.shape[0] == target_size[1], f"Height {resized.shape[0]} should equal target {target_size[1]}"
        
        # Check aspect ratio preservation
        if maintain_aspect:
            original_ratio = sample_cv2_image.shape[1] / sample_cv2_image.shape[0]
            resized_ratio = resized.shape[1] / resized.shape[0]
            assert abs(original_ratio - resized_ratio) < 0.1, "Aspect ratio should be preserved"
    
    @pytest.mark.parametrize('filter_type', ['blur', 'contour', 'detail', 'edge_enhance', 'sharpen'])
    def test_apply_filter_pil_image(self, sample_pil_image: Image.Image, filter_type: str):
        """Test applying various filters to PIL image"""
        filtered = ImageTransformer.apply_filter(sample_pil_image, filter_type=filter_type)
        assert isinstance(filtered, Image.Image), "Expected PIL Image"
        assert filtered.size == sample_pil_image.size, "Image dimensions should not change"
        assert filtered.mode == sample_pil_image.mode, "Color mode should not change"
    
    @pytest.mark.parametrize('filter_type', ['blur', 'gaussian_blur', 'median_blur'])
    def test_apply_filter_cv2_image(self, sample_cv2_image: np.ndarray, filter_type: str):
        """Test applying various filters to OpenCV image"""
        filtered = ImageTransformer.apply_filter(sample_cv2_image, filter_type=filter_type)
        assert isinstance(filtered, np.ndarray), "Expected numpy array"
        assert filtered.shape == sample_cv2_image.shape, "Image dimensions should not change"
        assert filtered.dtype == sample_cv2_image.dtype, "Data type should not change"
    
    @pytest.mark.parametrize('brightness,contrast', [
        (0.5, 1.0),    # Darker
        (1.5, 1.0),    # Brighter
        (1.0, 0.5),    # Less contrast
        (1.0, 1.5),    # More contrast
        (1.2, 1.2),    # Both increased
        (0.8, 0.8)     # Both decreased
    ])
    def test_adjust_brightness_contrast_pil_image(self, sample_pil_image: Image.Image, brightness: float, contrast: float):
        """Test brightness and contrast adjustment for PIL image"""
        adjusted = ImageTransformer.adjust_brightness_contrast(
            sample_pil_image, 
            brightness=brightness, 
            contrast=contrast
        )
        assert isinstance(adjusted, Image.Image), "Expected PIL Image"
        assert adjusted.size == sample_pil_image.size, "Image dimensions should not change"
        assert adjusted.mode == sample_pil_image.mode, "Color mode should not change"
    
    @pytest.mark.parametrize('brightness,contrast', [
        (0.5, 1.0),    # Darker
        (1.5, 1.0),    # Brighter
        (1.0, 0.5),    # Less contrast
        (1.0, 1.5),    # More contrast
        (1.2, 1.2),    # Both increased
        (0.8, 0.8)     # Both decreased
    ])
    def test_adjust_brightness_contrast_cv2_image(self, sample_cv2_image: np.ndarray, brightness: float, contrast: float):
        """Test brightness and contrast adjustment for OpenCV image"""
        adjusted = ImageTransformer.adjust_brightness_contrast(
            sample_cv2_image, 
            brightness=brightness, 
            contrast=contrast
        )
        assert isinstance(adjusted, np.ndarray), "Expected numpy array"
        assert adjusted.shape == sample_cv2_image.shape, "Image dimensions should not change"
        assert adjusted.dtype == sample_cv2_image.dtype, "Data type should not change"
    
    @pytest.mark.parametrize('size', [
        (128, 128),    # Default size
        (64, 64),      # Smaller
        (256, 256),    # Larger
        (150, 100)     # Non-square (matches original aspect ratio)
    ])
    def test_generate_thumbnail_pil_image(self, sample_pil_image: Image.Image, size: Tuple[int, int]):
        """Test thumbnail generation for PIL image with various sizes"""
        thumbnail = ImageTransformer.generate_thumbnail(sample_pil_image, size=size)
        assert isinstance(thumbnail, Image.Image), "Expected PIL Image"
        assert thumbnail.width <= size[0], f"Width {thumbnail.width} exceeds target {size[0]}"
        assert thumbnail.height <= size[1], f"Height {thumbnail.height} exceeds target {size[1]}"
        
        # Check aspect ratio preservation
        if size[0] == size[1]:  # Only check aspect ratio for square targets
            original_ratio = sample_pil_image.width / sample_pil_image.height
            thumbnail_ratio = thumbnail.width / thumbnail.height
            assert abs(original_ratio - thumbnail_ratio) < 0.1, "Aspect ratio should be preserved"
    
    @pytest.mark.parametrize('size', [
        (128, 128),    # Default size
        (64, 64),      # Smaller
        (256, 256),    # Larger
        (150, 100)     # Non-square (matches original aspect ratio)
    ])
    def test_generate_thumbnail_cv2_image(self, sample_cv2_image: np.ndarray, size: Tuple[int, int]):
        """Test thumbnail generation for OpenCV image with various sizes"""
        thumbnail = ImageTransformer.generate_thumbnail(sample_cv2_image, size=size)
        assert isinstance(thumbnail, np.ndarray), "Expected numpy array"
        assert thumbnail.shape[1] <= size[0], f"Width {thumbnail.shape[1]} exceeds target {size[0]}"
        assert thumbnail.shape[0] <= size[1], f"Height {thumbnail.shape[0]} exceeds target {size[1]}"
        
        # Check aspect ratio preservation
        if size[0] == size[1]:  # Only check aspect ratio for square targets
            original_ratio = sample_cv2_image.shape[1] / sample_cv2_image.shape[0]
            thumbnail_ratio = thumbnail.shape[1] / thumbnail.shape[0]
            assert abs(original_ratio - thumbnail_ratio) < 0.1, "Aspect ratio should be preserved"

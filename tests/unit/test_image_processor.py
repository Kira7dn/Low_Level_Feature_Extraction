import pytest
import io
import time
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2
from tests.constants import validate_response_structure, validate_processing_time
from app.services.image_processor import ImageProcessor

@pytest.fixture
def image_size() -> Tuple[int, int]:
    """Standard test image size"""
    return (100, 100)

@pytest.fixture
def large_image_size() -> Tuple[int, int]:
    """Large test image size for resize tests"""
    return (2000, 2000)

@pytest.fixture
def max_dimensions() -> Dict[str, int]:
    """Maximum dimensions for image processing"""
    return {
        'width': 1920,
        'height': 1080
    }

@pytest.fixture
def sample_pil_image(image_size: Tuple[int, int]) -> bytes:
    """Create a sample PIL image in PNG format"""
    img = Image.new('RGB', image_size, color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

@pytest.fixture
def large_pil_image(large_image_size: Tuple[int, int]) -> bytes:
    """Create a large PIL image for resize testing"""
    img = Image.new('RGB', large_image_size, color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

@pytest.fixture
def sample_cv2_image(image_size: Tuple[int, int]) -> bytes:
    """Create a sample OpenCV image in PNG format"""
    img = np.zeros((*image_size, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue in BGR format
    success, img_bytes = cv2.imencode('.png', img)
    assert success, "Failed to encode OpenCV image"
    return img_bytes.tobytes()

@pytest.fixture
def large_cv2_image(large_image_size: Tuple[int, int]) -> bytes:
    """Create a large OpenCV image for resize testing"""
    img = np.zeros((*large_image_size, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue in BGR format
    success, img_bytes = cv2.imencode('.png', img)
    assert success, "Failed to encode OpenCV image"
    return img_bytes.tobytes()

@pytest.fixture
def cdn_config() -> Dict[str, Any]:
    """Default CDN configuration"""
    return {
        'base_url': 'https://cdn.example.com/images',
        'image_path': '/path/to/sample.jpg'
    }

class TestImageProcessor:

    def test_load_pil_image(self, sample_pil_image: bytes, image_size: Tuple[int, int]):
        """Test loading a PIL image from bytes"""
        # Load and validate image
        start_time = time.time()
        image = ImageProcessor.load_image(sample_pil_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": image}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": Image.Image},
            context="image_processing"
        )
        assert is_valid, error_msg
        assert result["image"].size == image_size
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg

    def test_load_cv2_image(self, sample_cv2_image: bytes, image_size: Tuple[int, int]):
        """Test loading an OpenCV image from bytes"""
        # Load and validate image
        start_time = time.time()
        image = ImageProcessor.load_cv2_image(sample_cv2_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": image}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": np.ndarray},
            context="image_processing"
        )
        assert is_valid, error_msg
        assert result["image"].shape[:2] == image_size[::-1]  # OpenCV uses (height, width) order
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg

    @pytest.mark.parametrize('target_size', [
        (100, 75),  # Downscale
        (30, 40),   # Smaller dimensions
        (150, 200)  # Upscale
    ])
    def test_resize_pil_image(self, sample_pil_image: bytes, target_size: Tuple[int, int]):
        """Test PIL image resizing with various target sizes"""
        # Load and validate image
        start_time = time.time()
        image = ImageProcessor.load_image(sample_pil_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": image}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": Image.Image},
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Resize and validate image
        start_time = time.time()
        resized = ImageProcessor.resize_image(image, *target_size)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": resized}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": Image.Image},
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Check that dimensions are within the target bounds
        assert resized.size[0] <= target_size[0], f"Width {resized.size[0]} exceeds target {target_size[0]}"
        assert resized.size[1] <= target_size[1], f"Height {resized.size[1]} exceeds target {target_size[1]}"
        # Check aspect ratio is preserved (with small tolerance for rounding)
        original_ratio = image.size[0] / image.size[1]
        resized_ratio = resized.size[0] / resized.size[1]
        assert abs(original_ratio - resized_ratio) < 0.1, "Aspect ratio should be preserved"

    @pytest.mark.parametrize('target_size', [
        (100, 75),  # Downscale
        (30, 40),   # Smaller dimensions
        (150, 200)  # Upscale
    ])
    def test_resize_cv2_image(self, sample_cv2_image: bytes, target_size: Tuple[int, int]):
        """Test OpenCV image resizing with various target sizes"""
        # Load and validate image
        start_time = time.time()
        image = ImageProcessor.load_cv2_image(sample_cv2_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": image}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": np.ndarray},
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Resize and validate image
        start_time = time.time()
        resized = ImageProcessor.resize_image(image, *target_size)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image": resized}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image"],
            value_types={"image": np.ndarray},
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg
        
        # Check that dimensions are within the target bounds
        assert resized.shape[0] <= target_size[1], f"Height {resized.shape[0]} exceeds target {target_size[1]}"
        assert resized.shape[1] <= target_size[0], f"Width {resized.shape[1]} exceeds target {target_size[0]}"
        # Check aspect ratio is preserved (with small tolerance for rounding)
        original_ratio = image.shape[1] / image.shape[0]  # width/height for OpenCV
        resized_ratio = resized.shape[1] / resized.shape[0]
        assert abs(original_ratio - resized_ratio) < 0.1, "Aspect ratio should be preserved"

    # CDN URL generation tests will be implemented later

    def test_auto_process_image_pil(
        self,
        large_pil_image: bytes,
        max_dimensions: Dict[str, int]
    ):
        """Test automatic image processing with large PIL image"""
        # Process and validate image
        start_time = time.time()
        processed_bytes = ImageProcessor.auto_process_image(
            large_pil_image,
            max_dimensions['width'],
            max_dimensions['height']
        )
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image_bytes": processed_bytes}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image_bytes"],
            value_types={"image_bytes": bytes},
            context="image_processing"
        )
        assert is_valid, error_msg
        assert len(processed_bytes) > 0
        
        # Verify dimensions
        img = Image.open(io.BytesIO(processed_bytes))
        assert img.width <= max_dimensions['width']
        assert img.height <= max_dimensions['height']
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg

    def test_auto_process_image_cv2(
        self,
        large_cv2_image: bytes,
        max_dimensions: Dict[str, int]
    ):
        """Test automatic image processing with large OpenCV image"""
        # Process and validate image
        start_time = time.time()
        processed_bytes = ImageProcessor.auto_process_image(
            large_cv2_image,
            max_dimensions['width'],
            max_dimensions['height']
        )
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        result = {"image_bytes": processed_bytes}
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["image_bytes"],
            value_types={"image_bytes": bytes},
            context="image_processing"
        )
        assert is_valid, error_msg
        assert len(processed_bytes) > 0
        
        # Verify dimensions
        img = Image.open(io.BytesIO(processed_bytes))
        assert img.width <= max_dimensions['width']
        assert img.height <= max_dimensions['height']
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context="image_processing"
        )
        assert is_valid, error_msg

    @pytest.mark.parametrize('target_format', ['jpeg', 'webp', 'png'])
    def test_convert_format_pil(self, sample_pil_image: bytes, target_format: str):
        """Test format conversion for PIL images"""
        image = ImageProcessor.load_image(sample_pil_image)
        converted = ImageProcessor.convert_format(image, target_format=target_format)
        
        assert isinstance(converted, bytes), "Expected bytes output"
        
        # Verify format conversion
        img = Image.open(io.BytesIO(converted))
        assert img.format == target_format.upper(), \
            f"Expected format {target_format.upper()}, got {img.format}"

    @pytest.mark.parametrize('target_format', ['jpeg', 'webp', 'png'])
    def test_convert_format_cv2(self, sample_cv2_image: bytes, target_format: str):
        """Test format conversion for OpenCV images"""
        image = ImageProcessor.load_cv2_image(sample_cv2_image)
        converted = ImageProcessor.convert_format(image, target_format=target_format)
        
        assert isinstance(converted, bytes), "Expected bytes output"
        
        # Verify format conversion
        img = Image.open(io.BytesIO(converted))
        assert img.format == target_format.upper(), \
            f"Expected format {target_format.upper()}, got {img.format}"
        
        # Verify image can be decoded back to OpenCV format
        cv_img = cv2.imdecode(np.frombuffer(converted, np.uint8), cv2.IMREAD_COLOR)
        assert cv_img is not None, "Failed to decode converted image"
        assert cv_img.shape == image.shape, "Image dimensions should be preserved"

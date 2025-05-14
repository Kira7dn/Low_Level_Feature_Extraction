import pytest
import io
from PIL import Image
import numpy as np
import cv2

from app.services.image_transformer import ImageTransformer

@pytest.fixture
def sample_pil_image():
    # Create a simple PIL image
    img = Image.new('RGB', (200, 200), color='red')
    return img

@pytest.fixture
def sample_cv2_image():
    # Create a simple OpenCV image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue image
    return img

def test_resize_pil_image(sample_pil_image):
    # Test resizing PIL image with aspect ratio
    resized = ImageTransformer.resize(sample_pil_image, width=100)
    assert isinstance(resized, Image.Image)
    assert resized.width <= 100
    assert resized.height <= 200

    # Test resizing without maintaining aspect ratio
    resized = ImageTransformer.resize(sample_pil_image, width=50, height=50, maintain_aspect_ratio=False)
    assert resized.width == 50
    assert resized.height == 50

def test_resize_cv2_image(sample_cv2_image):
    # Test resizing OpenCV image with aspect ratio
    resized = ImageTransformer.resize(sample_cv2_image, width=100)
    assert isinstance(resized, np.ndarray)
    assert resized.shape[1] <= 100
    assert resized.shape[0] <= 200

    # Test resizing without maintaining aspect ratio
    resized = ImageTransformer.resize(sample_cv2_image, width=50, height=50, maintain_aspect_ratio=False)
    assert resized.shape[1] == 50
    assert resized.shape[0] == 50

def test_apply_filter_pil_image(sample_pil_image):
    # Test various PIL filters
    filters = ['blur', 'contour', 'detail', 'edge_enhance', 'emboss', 'sharpen']
    
    for filter_type in filters:
        filtered = ImageTransformer.apply_filter(sample_pil_image, filter_type)
        assert isinstance(filtered, Image.Image)

def test_apply_filter_cv2_image(sample_cv2_image):
    # Test various OpenCV filters
    filters = ['blur', 'gaussian_blur', 'median_blur']
    
    for filter_type in filters:
        filtered = ImageTransformer.apply_filter(sample_cv2_image, filter_type)
        assert isinstance(filtered, np.ndarray)

def test_adjust_brightness_contrast_pil_image(sample_pil_image):
    # Test brightness and contrast adjustment for PIL image
    adjusted = ImageTransformer.adjust_brightness_contrast(
        sample_pil_image, 
        brightness=1.5, 
        contrast=1.2
    )
    assert isinstance(adjusted, Image.Image)

def test_adjust_brightness_contrast_cv2_image(sample_cv2_image):
    # Test brightness and contrast adjustment for OpenCV image
    adjusted = ImageTransformer.adjust_brightness_contrast(
        sample_cv2_image, 
        brightness=1.5, 
        contrast=1.2
    )
    assert isinstance(adjusted, np.ndarray)

def test_generate_thumbnail_pil_image(sample_pil_image):
    # Test thumbnail generation for PIL image
    thumbnail = ImageTransformer.generate_thumbnail(sample_pil_image)
    assert isinstance(thumbnail, Image.Image)
    assert thumbnail.width <= 128
    assert thumbnail.height <= 128

def test_generate_thumbnail_cv2_image(sample_cv2_image):
    # Test thumbnail generation for OpenCV image
    thumbnail = ImageTransformer.generate_thumbnail(sample_cv2_image)
    assert isinstance(thumbnail, np.ndarray)
    assert thumbnail.shape[1] <= 128
    assert thumbnail.shape[0] <= 128

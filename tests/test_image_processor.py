import pytest
import io
from PIL import Image
import numpy as np
import cv2

from app.services.image_processor import ImageProcessor

def test_load_pil_image():
    # Create a simple PIL image
    img = Image.new('RGB', (100, 100), color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    sample_pil_image = byte_arr.getvalue()

    image = ImageProcessor.load_image(sample_pil_image)
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)

def test_load_cv2_image():
    # Create a simple OpenCV image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue image
    sample_cv2_image = cv2.imencode('.png', img)[1].tobytes()

    image = ImageProcessor.load_cv2_image(sample_cv2_image)
    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 100, 3)

def test_resize_pil_image():
    # Create a simple PIL image
    img = Image.new('RGB', (100, 100), color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    sample_pil_image = byte_arr.getvalue()

    image = ImageProcessor.load_image(sample_pil_image)
    resized = ImageProcessor.resize_image(image, max_width=50, max_height=50)
    assert isinstance(resized, Image.Image)
    assert resized.size == (50, 50)

def test_resize_cv2_image():
    # Create a simple OpenCV image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue image
    sample_cv2_image = cv2.imencode('.png', img)[1].tobytes()

    image = ImageProcessor.load_cv2_image(sample_cv2_image)
    resized = ImageProcessor.resize_image(image, max_width=50, max_height=50)
    assert isinstance(resized, np.ndarray)
    assert resized.shape[:2] == (50, 50)

def test_generate_cdn_url_default():
    """Test default CDN URL generation"""
    base_url = 'https://cdn.example.com/images'
    image_path = '/path/to/sample.jpg'
    
    cdn_url = ImageProcessor.generate_cdn_url(base_url, image_path)
    
    # Verify base URL structure
    assert cdn_url.startswith(base_url)
    assert 'sample.jpg' in cdn_url
    
    # Verify default transformations
    assert 'f:webp' in cdn_url
    assert 'q:85' in cdn_url
    assert 'w:1920,h:1080,fit:max' in cdn_url

def test_generate_cdn_url_custom_transformations():
    """Test CDN URL generation with custom transformations"""
    base_url = 'https://cdn.example.com/images'
    image_path = '/path/to/custom.png'
    
    cdn_url = ImageProcessor.generate_cdn_url(
        base_url, 
        image_path, 
        transformations={
            'format': 'avif',
            'quality': 75,
            'resize': {
                'width': 800,
                'height': 600,
                'fit': 'crop'
            }
        }
    )
    
    # Verify base URL structure
    assert cdn_url.startswith(base_url)
    assert 'custom.png' in cdn_url
    
    # Verify custom transformations
    assert 'f:avif' in cdn_url
    assert 'q:75' in cdn_url
    assert 'w:800,h:600,fit:crop' in cdn_url

def test_generate_cdn_url_invalid_inputs():
    """Test CDN URL generation with invalid inputs"""
    with pytest.raises(ValueError, match="Base URL and image path must be provided"):
        ImageProcessor.generate_cdn_url('', '')
    
    with pytest.raises(ValueError, match="Base URL and image path must be provided"):
        ImageProcessor.generate_cdn_url(None, None)

def test_auto_process_image_pil():
    """Test automatic image processing with PIL image"""
    # Create a simple PIL image
    img = Image.new('RGB', (2000, 2000), color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    sample_pil_image = byte_arr.getvalue()

    # Process the image
    processed_image = ImageProcessor.auto_process_image(
        sample_pil_image, 
        max_width=1920, 
        max_height=1080, 
        target_format='webp'
    )

    # Verify the processed image
    assert isinstance(processed_image, bytes)
    
    # Verify image properties
    processed_img = Image.open(io.BytesIO(processed_image))
    assert processed_img.format == 'WEBP'
    assert processed_img.width <= 1920
    assert processed_img.height <= 1080

def test_auto_process_image_cv2():
    """Test automatic image processing with OpenCV image"""
    # Create a large OpenCV image
    img = np.zeros((2500, 2500, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue image
    sample_cv2_image = cv2.imencode('.png', img)[1].tobytes()

    # Process the image
    processed_image = ImageProcessor.auto_process_image(
        sample_cv2_image, 
        max_width=1920, 
        max_height=1080, 
        target_format='webp'
    )

    # Verify the processed image
    assert isinstance(processed_image, bytes)
    
    # Verify image properties
    processed_img = Image.open(io.BytesIO(processed_image))
    assert processed_img.format == 'WEBP'
    assert processed_img.width <= 1920
    assert processed_img.height <= 1080

def test_convert_format_pil():
    # Create a simple PIL image
    img = Image.new('RGB', (100, 100), color='red')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    sample_pil_image = byte_arr.getvalue()

    image = ImageProcessor.load_image(sample_pil_image)
    converted = ImageProcessor.convert_format(image, target_format='jpeg')
    assert isinstance(converted, bytes)
    
    # Verify it's a valid JPEG
    img = Image.open(io.BytesIO(converted))
    assert img.format == 'JPEG'

def test_convert_format_cv2():
    # Create a simple OpenCV image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = [0, 0, 255]  # Blue image
    sample_cv2_image = cv2.imencode('.png', img)[1].tobytes()

    image = ImageProcessor.load_cv2_image(sample_cv2_image)
    converted = ImageProcessor.convert_format(image, target_format='jpeg')
    assert isinstance(converted, bytes)
    
    # Verify it's a valid JPEG
    img = Image.open(io.BytesIO(converted))
    assert img.format == 'JPEG'

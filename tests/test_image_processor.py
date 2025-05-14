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

import sys
import os
import base64
import io
import pytest
import asyncio
from typing import Dict, Union, List
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import numpy as np

# Ensure absolute imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the main app
import importlib.util
spec = importlib.util.spec_from_file_location(
    'app.main', 
    os.path.join(project_root, 'app', 'main.py')
)
module = importlib.util.module_from_spec(spec)
sys.modules['app.main'] = module
spec.loader.exec_module(module)

app = module.app
client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image with known colors"""
    # Create an image with a red background and blue rectangle
    img = Image.new('RGB', (300, 200), color='red')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 150], fill='blue')
    
    # Save to byte array
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr

@pytest.fixture
def sample_image_base64(sample_image):
    """Convert sample image to base64"""
    return base64.b64encode(sample_image.getvalue()).decode('utf-8')

@pytest.fixture
def test_image_file(sample_image):
    from fastapi import UploadFile
    from io import BytesIO
    return UploadFile(filename="test_image.png", file=sample_image)

@pytest.fixture
def test_base64_image(sample_image_base64):
    return sample_image_base64

def test_color_extraction_file_upload(sample_image):
    """Test color extraction with file upload"""
    # Prepare the file for upload
    sample_image.seek(0)
    response = client.post("/colors/extract", files={"file": ("test_image.png", sample_image, "image/png")}, params={"n_colors": 5})
    
    # Check response
    assert response.status_code == 200, f"Response content: {response.text}"
    
    # Validate response structure
    data = response.json()
    assert "colors" in data
    assert "primary_color" in data
    assert "background_color" in data
    
    # Validate color format
    assert len(data["colors"]) > 0
    for color in data["colors"]:
        assert "hex" in color
        assert "rgb" in color
        assert color["hex"].startswith("#")
        assert len(color["rgb"]) == 3

@pytest.mark.asyncio
async def test_extract_colors_base64(test_base64_image, n_colors=5):
    # The fixture now returns the base64 image directly
    base64_image = test_base64_image
    
    print(f"Base64 Image Length: {len(base64_image)}")
    print(f"Base64 Image Sample: {base64_image[:100]}")
    
    # Ensure proper base64 padding
    padded_image = base64_image
    if len(padded_image) % 4 != 0:
        padded_image += '=' * (4 - (len(padded_image) % 4))
    
    response = client.post("/colors/extract-base64", json={"base64_image": padded_image}, params={"n_colors": n_colors})
    
    # Check response
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
    assert response.status_code == 200, f"Response content: {response.text}"
    
    # Validate response structure
    data = response.json()
    assert "colors" in data
    assert "primary_color" in data
    assert "background_color" in data
    
    # Validate color format
    assert len(data["colors"]) > 0
    for color in data["colors"]:
        assert "hex" in color
        assert "rgb" in color
        assert color["hex"].startswith("#")
        assert len(color["rgb"]) == 3

async def test_extract_colors(test_image_file, n_colors=5):
    """Test color extraction with file upload"""
    # Prepare the file for upload
    test_image_file.file.seek(0)

    # Create a file-like object for upload
    from io import BytesIO
    file_like = BytesIO(test_image_file.file.read())

    response = client.post("/colors/extract", files={"file": (test_image_file.filename, file_like, test_image_file.content_type)}, params={"n_colors": n_colors})
    
    # Check response
    assert response.status_code == 200, f"Response content: {response.text}"
    
    # Validate response structure
    data = response.json()
    assert "colors" in data
    assert "primary_color" in data
    assert "background_color" in data
    
    # Validate color format
    assert len(data["colors"]) > 0
    for color in data["colors"]:
        assert "hex" in color
        assert "rgb" in color
        assert color["hex"].startswith("#")
        assert len(color["rgb"]) == 3

def test_color_extractor_service():
    """Test the ColorExtractor service directly"""
    from app.services.color_extractor import ColorExtractor
    
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Extract colors
    colors = ColorExtractor.extract_colors(img)
    
    # Validate results
    assert len(colors) > 0
    assert all(color.startswith("#") for color in colors)
    
    # Test palette analysis
    palette = ColorExtractor.analyze_palette(img)
    
    assert "colors" in palette
    assert "primary_color" in palette
    assert "background_color" in palette
    
    # Validate color details
    assert len(palette["colors"]) > 0
    for color in palette["colors"]:
        assert "hex" in color
        assert "rgb" in color
        assert "name" in color
        assert color["hex"].startswith("#")
        assert len(color["rgb"]) == 3
        assert isinstance(color["name"], str)
        assert len(color["name"]) > 0

import io
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import app

def create_synthetic_font_image(intensity=200, size=(300, 300)):
    """
    Create a synthetic image with a text-like region for font detection testing
    
    Args:
        intensity (int): Pixel intensity for the text region
        size (tuple): Image dimensions
    
    Returns:
        io.BytesIO: Image bytes for upload
    """
    # Create blank image
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    
    # Create text-like region
    x, y = size[0] // 4, size[1] // 4
    w, h = size[0] // 2, size[1] // 2
    cv2.rectangle(
        img, 
        (x, y), 
        (x + w, y + h), 
        (intensity, intensity, intensity), 
        -1
    )
    
    # Convert to PNG bytes
    _, buf = cv2.imencode('.png', img)
    return io.BytesIO(buf.tobytes())

def test_font_endpoint_light_font():
    """Test font detection endpoint with a light font image"""
    client = TestClient(app)
    img_bytes = create_synthetic_font_image(intensity=240)
    
    response = client.post(
        "/fonts/extract-fonts",
        files={"file": ("font.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    font_data = response.json()
    
    # Validate response structure
    assert "font_family" in font_data
    assert "font_size" in font_data
    assert "font_weight" in font_data
    
    # Validate specific values
    assert font_data["font_weight"] == "Regular"
    assert isinstance(font_data["font_size"], int)
    assert font_data["font_size"] > 0

def test_font_endpoint_regular_font():
    """Test font detection endpoint with a regular font image"""
    client = TestClient(app)
    img_bytes = create_synthetic_font_image(intensity=200)
    
    response = client.post(
        "/fonts/extract-fonts",
        files={"file": ("font.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    font_data = response.json()
    
    # Validate response structure
    assert "font_family" in font_data
    assert "font_size" in font_data
    assert "font_weight" in font_data
    
    # Validate specific values
    assert font_data["font_weight"] == "Regular"
    assert isinstance(font_data["font_size"], int)
    assert font_data["font_size"] > 0

def test_font_endpoint_bold_font():
    """Test font detection endpoint with a bold font image"""
    client = TestClient(app)
    img_bytes = create_synthetic_font_image(intensity=50)
    
    response = client.post(
        "/fonts/extract-fonts",
        files={"file": ("font.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    font_data = response.json()
    
    # Validate response structure
    assert "font_family" in font_data
    assert "font_size" in font_data
    assert "font_weight" in font_data
    
    # Validate specific values
    assert font_data["font_weight"] == "Bold"
    assert isinstance(font_data["font_size"], int)
    assert font_data["font_size"] > 0

def test_font_endpoint_no_text():
    """Test font detection endpoint with an empty/blank image"""
    client = TestClient(app)
    img_bytes = create_synthetic_font_image(intensity=255)
    
    response = client.post(
        "/fonts/extract-fonts",
        files={"file": ("font.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    font_data = response.json()
    
    # Validate response for no text
    assert font_data["font_family"] == "Unknown"
    assert font_data["font_size"] == 0
    assert font_data["font_weight"] == "Unknown"

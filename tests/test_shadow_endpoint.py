import io
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import app

def create_shadow_image_bytes(intensity=50, size=(100, 100)):
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (intensity, intensity, intensity), -1)
    _, buf = cv2.imencode('.png', img)
    return io.BytesIO(buf.tobytes())

def test_shadow_endpoint_high():
    client = TestClient(app)
    img_bytes = create_shadow_image_bytes(intensity=30)
    response = client.post(
        "/extract-shadows",
        files={"file": ("shadow.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["shadow_level"] == "High"

def test_shadow_endpoint_moderate():
    client = TestClient(app)
    img_bytes = create_shadow_image_bytes(intensity=210)
    response = client.post(
        "/extract-shadows",
        files={"file": ("shadow.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["shadow_level"] == "Moderate"

def test_shadow_endpoint_low():
    client = TestClient(app)
    img_bytes = create_shadow_image_bytes(intensity=240)
    response = client.post(
        "/extract-shadows",
        files={"file": ("shadow.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["shadow_level"] == "Low"

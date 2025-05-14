import cv2
import numpy as np
import pytest
from app.services.shadow_analyzer import ShadowAnalyzer

# Helper to create a synthetic image with a shadow

def create_shadow_image(intensity=50, size=(100, 100)):
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    # Add a rectangle shadow region
    cv2.rectangle(img, (30, 30), (70, 70), (intensity, intensity, intensity), -1)
    return img

def test_no_shadow():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = ShadowAnalyzer.analyze_shadow_level(img)
    assert result == 'Low'

def test_high_shadow():
    img = create_shadow_image(intensity=30)
    result = ShadowAnalyzer.analyze_shadow_level(img)
    assert result == 'High'

def test_moderate_shadow():
    # Use a value between the thresholds for Moderate (e.g., intensity=210)
    img = create_shadow_image(intensity=210)
    result = ShadowAnalyzer.analyze_shadow_level(img)
    assert result == 'Moderate'

def test_low_shadow():
    img = create_shadow_image(intensity=240)
    result = ShadowAnalyzer.analyze_shadow_level(img)
    assert result == 'Low'

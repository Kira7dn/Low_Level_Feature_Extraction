import os
import subprocess
import sys
import pytest

def test_project_structure():
    """Verify project structure is correctly set up"""
    # Check required directories exist
    required_dirs = [
        'app',
        'app/services',
        'app/routers',
        'app/utils',
        'tests'
    ]
    
    for directory in required_dirs:
        assert os.path.exists(directory), f"Directory {directory} is missing"

def test_dependencies():
    """Ensure all dependencies are installed correctly"""
    # Run pip list and check for key dependencies
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list'], 
        capture_output=True, 
        text=True
    )
    
    required_packages = [
        'fastapi',
        'opencv-python',
        'numpy',
        'uvicorn',
        'pytest'
    ]
    
    for package in required_packages:
        assert package in result.stdout, f"{package} is not installed"

def test_application_startup():
    """Test that the application can start without runtime exceptions"""
    try:
        import uvicorn
        from app.main import app
    except Exception as e:
        pytest.fail(f"Failed to import application: {e}")

def test_root_endpoint():
    """Verify root endpoint functionality"""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]

def test_health_endpoint():
    """Perform basic health check"""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_shadows_endpoint_response_format():
    """Validate shadows endpoint response format"""
    from fastapi.testclient import TestClient
    from app.main import app
    import cv2
    import numpy as np
    
    # Create a test image
    image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (100, 100), (250, 200), (150, 150, 150), -1)
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_shadow_image.png')
    cv2.imwrite(test_image_path, image)
    
    client = TestClient(app)
    
    with open(test_image_path, 'rb') as f:
        response = client.post(
            "/extract-shadows",
            files={"file": ("test_image.png", f, "image/png")}
        )
    
    # Validate response structure
    assert response.status_code == 200
    data = response.json()
    
    # Check keys exist
    assert "shadow_level" in data
    
    # Validate value types and ranges
    assert data["shadow_level"] in ["Low", "Medium", "High", "none"]

def test_performance():
    """Test endpoint response time"""
    from fastapi.testclient import TestClient
    from app.main import app
    import cv2
    import numpy as np
    import time
    
    # Create a moderately complex test image
    image = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
    for i in range(5):
        cv2.rectangle(
            image, 
            (i*200, i*200), 
            ((i+1)*200, (i+1)*200), 
            (50+i*30, 50+i*30, 50+i*30), 
            -1
        )
    test_image_path = os.path.join(os.path.dirname(__file__), 'performance_test_image.png')
    cv2.imwrite(test_image_path, image)
    
    client = TestClient(app)
    
    start_time = time.time()
    with open(test_image_path, 'rb') as f:
        response = client.post(
            "/extract-shadows",
            files={"file": ("performance_test.png", f, "image/png")}
        )
    end_time = time.time()
    
    # Verify response
    assert response.status_code == 200
    
    # Check response time
    response_time = end_time - start_time
    assert response_time < 1.0, f"Response time too slow: {response_time} seconds"

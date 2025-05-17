"""Test image upload functionality to the /api/v1/analyze endpoint."""
import os
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Get the absolute path to the test image
TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_data",
    "test_suite",
    "simple_text.png"
)

client = TestClient(app)

def test_image_upload():
    """Test that an image can be successfully uploaded to the analyze endpoint."""
    print("\n=== STARTING IMAGE UPLOAD TEST ===")
    
    # Verify the test image exists
    print(f"Checking if test image exists at: {TEST_IMAGE_PATH}")
    assert os.path.exists(TEST_IMAGE_PATH), f"Test image not found at {TEST_IMAGE_PATH}"
    print("✓ Test image found")
    
    # Read the image file
    with open(TEST_IMAGE_PATH, 'rb') as f:
        image_data = f.read()
    
    # Verify the image data
    print(f"Read {len(image_data)} bytes from image file")
    assert len(image_data) > 0, "Image file is empty"
    
    # Check if it looks like a PNG file
    is_png = image_data.startswith(b'\x89PNG\r\n\x1a\n')
    print(f"Image appears to be a valid PNG: {is_png}")
    
    # Prepare the multipart form data
    files = {
        'file': ('test_image.png', image_data, 'image/png')
    }
    
    # Additional form data
    form_data = {
        'preprocessing': 'auto',
        'features': ['colors', 'text', 'fonts']
    }
    
    print("\nSending request to /api/v1/analyze")
    print(f"Files: {list(files.keys())}")
    print(f"Form data: {form_data}")
    
    try:
        # Make the request
        response = client.post(
            "/api/v1/analyze",
            files=files,
            data=form_data
        )
        
        # Print response information
        print("\n=== RESPONSE ===")
        print(f"Status code: {response.status_code}")
        print("Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
            
        # Try to parse JSON response
        try:
            json_response = response.json()
            print("\nResponse JSON:")
            print(json.dumps(json_response, indent=2))
            
            # Basic assertions
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
            assert 'features' in json_response, "Response missing 'features' key"
            
            # Check if we got partial success
            if json_response.get('status') == 'partial':
                print("\nPartial success - some features may have failed:")
                if 'errors' in json_response:
                    for feature, error in json_response['errors'].items():
                        print(f"  {feature}: {error['message']}")
            
            return json_response
            
        except ValueError as e:
            print(f"Failed to parse response as JSON: {e}")
            print(f"Raw response: {response.text}")
            raise
            
    except Exception as e:
        print(f"Error making request: {str(e)}")
        raise
    
    try:
        response_data = response.json()
        print("Response JSON:", response_data)
    except Exception as e:
        print(f"Failed to parse response as JSON: {e}")
        print(f"Raw response: {response.text}")
    
    # Basic assertions
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    assert "features" in response.json(), "Response should contain 'features' key"

if __name__ == "__main__":
    test_image_upload()

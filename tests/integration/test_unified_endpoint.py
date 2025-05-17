"""Integration tests for the unified analysis endpoint."""
import json
import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient

from app.main import app
from tests.constants import (
    ValidationRules,
    validate_response_structure,
    validate_processing_time,
    get_performance_rating
)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_SUITE_PATH = PROJECT_ROOT / "tests" / "test_data" / "test_suite" / "test_suite.json"

# Initialize test client
client = TestClient(app)

# Test configuration
MAX_PROCESSING_TIME = 5.0  # seconds

def load_test_suite() -> list[dict]:
    """Load and return test cases from the test suite JSON file."""
    if not TEST_SUITE_PATH.exists():
        pytest.skip(f"Test suite not found: {TEST_SUITE_PATH}")
    
    try:
        with open(TEST_SUITE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('test_cases', [])
    except (json.JSONDecodeError, OSError) as e:
        pytest.fail(f"Error loading test suite: {str(e)}")


def compare_colors(color1: str, color2: str, threshold: int = 10) -> bool:
    """Compare two hex colors with a given threshold."""
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    return all(abs(c1 - c2) <= threshold for c1, c2 in zip(rgb1, rgb2))


class TestUnifiedAnalysisEndpoint:
    """Integration tests for the unified analysis endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment."""
        self.test_cases = load_test_suite()
    
    def _get_full_image_path(self, image_path: str) -> str:
        """Convert relative image path to absolute path.
        
        Args:
            image_path: Path to the image, either absolute or relative to the test_suite.json
            
        Returns:
            str: Absolute path to the image file
            
        Raises:
            FileNotFoundError: If the image file cannot be found
        """
        path = Path(image_path)
        
        # If path is already absolute, return it as is
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            return str(path)
            
        # Try to find the image relative to the test_suite.json location
        test_suite_dir = TEST_SUITE_PATH.parent
        test_image_path = test_suite_dir / path
        
        # If not found, try relative to the tests directory
        if not test_image_path.exists():
            test_image_path = PROJECT_ROOT / 'tests' / path
            
        if test_image_path.exists():
            return str(test_image_path)
            
        # If still not found, try relative to project root as last resort
        root_path = PROJECT_ROOT / path
        if root_path.exists():
            return str(root_path)
            
        raise FileNotFoundError(
            f"Image file not found: {path}. "
            f"Tried: {test_suite_dir / path}, {PROJECT_ROOT / 'tests' / path}, {root_path}"
        )
    
    def _test_feature_match(self, expected: Any, actual: Any, feature_name: str):
        """Helper to test if expected feature matches actual result.
        
        Args:
            expected: Dictionary of expected feature values
            actual: Dictionary of actual feature values from the API
            feature_name: Name of the feature being tested (colors, text, fonts)
            
        Raises:
            AssertionError: If the actual values don't match the expected values
        """
        if feature_name == 'colors':
            # Check if required keys exist
            required_keys = ['primary', 'background', 'accent']
            for key in required_keys:
                assert key in expected, f"Expected key '{key}' not found in expected features"
                assert key in actual, f"Key '{key}' not found in actual features"
            
            # Test primary color
            if expected['primary'] and actual['primary']:
                assert compare_colors(
                    expected['primary'].lower(), 
                    actual['primary'].lower()
                ), f"Primary color mismatch. Expected {expected['primary']}, got {actual['primary']}"
            
            # Test background color
            if expected['background'] and actual['background']:
                assert compare_colors(
                    expected['background'].lower(),
                    actual['background'].lower()
                ), f"Background color mismatch. Expected {expected['background']}, got {actual['background']}"
            
            # Test accent colors (check if expected accents are in actual accents)
            if expected.get('accent') and actual.get('accent'):
                for exp_accent in expected['accent']:
                    assert any(
                        compare_colors(exp_accent.lower(), act_accent.lower())
                        for act_accent in actual['accent']
                    ), f"Accent color {exp_accent} not found in {actual['accent']}"
                
        elif feature_name == 'text':
            # Check if all expected lines are present in the actual lines
            for exp_line in expected['lines']:
                assert any(
                    exp_line.lower() in act_line.lower()
                    for act_line in actual['lines']
                ), f"Expected text not found: {exp_line}"
                
        elif feature_name == 'fonts':
            # Check font family (case insensitive)
            if 'font_family' in expected and expected['font_family']:
                assert 'font_family' in actual, "Font family not found in response"
                assert actual['font_family'].lower() == expected['font_family'].lower(), \
                    f"Font family mismatch. Expected {expected['font_family']}, got {actual['font_family']}"
    
    def get_test_cases():
        """Get test cases and print their names for debugging."""
        test_cases = load_test_suite()
        print("\nAvailable test cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"{i}. {case['name']} - {case['description']}")
            print(f"   Image path: {case['image_path']}")
            print(f"   Expected features: {list(case['expected_features'].keys())}")
        return test_cases
    
    @pytest.mark.parametrize("test_case", get_test_cases(), 
                           ids=lambda x: x.get('name', 'unnamed'))
    def test_analyze_image(self, test_case: Dict[str, Any]):
        """Test the unified analysis endpoint with test suite images."""
        print(f"\n{'='*80}\nRunning test case: {test_case['name']}")
        print(f"Description: {test_case['description']}")

        # Get the full path to the test image
        image_path = self._get_full_image_path(test_case['image_path'])
        print(f"Image path: {image_path}")
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        # Verify the image can be loaded by OpenCV
        try:
            img = cv2.imread(image_path)
            assert img is not None, "Failed to load image with OpenCV"
            print(f"Image loaded successfully. Shape: {img.shape}")
        except Exception as e:
            print(f"Error loading image with OpenCV: {str(e)}")
            # Try to read the file as bytes to verify it's not empty
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
                print(f"Image file size: {len(img_bytes)} bytes")
                assert len(img_bytes) > 0, "Image file is empty"
            raise

        # Prepare the multipart form data with file and form fields
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            print(f"Read {len(img_data)} bytes from image file")

            # Create a proper file upload object with form data
            files = {
                'file': (os.path.basename(image_path), img_data, 'image/png'),
            }

            # Add form fields
            data = {
                'preprocessing': 'auto',
                'features': ['colors', 'text', 'fonts']
            }

            print(f"Sending request to /api/v1/analyze with features: colors, text, fonts")
            print(f"Image file: {image_path}")

            # Make the request to the endpoint with both files and form data
            response = client.post(
                "/api/v1/analyze",
                files=files,
                data=data
            )
            print(f"Response status: {response.status_code}")

            # Print response content for debugging
            try:
                response_data = response.json()
                print("Response content:", json.dumps(response_data, indent=2))
            except Exception as e:
                print(f"Failed to parse response as JSON: {str(e)}")
                print(f"Raw response: {response.text}")
                raise
            
            # Check response status code
            assert response.status_code == 200, \
                f"Request failed with status {response.status_code}: {response.text}"
        
        # Validate processing time
        processing_time = response.elapsed.total_seconds()
        assert validate_processing_time(processing_time, MAX_PROCESSING_TIME), \
            f"Processing time {processing_time}s exceeds maximum limit"
        
        # Log performance rating
        performance_rating = get_performance_rating(processing_time)
        print(f"Performance Rating: {performance_rating}")
        
        # Parse response
        result = response.json()
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["status", "features", "errors", "metadata"],
            value_types={
                "status": str,
                "features": dict,
                "errors": dict,
                "metadata": dict
            }
        )
        assert is_valid, f"Invalid response structure: {error_msg}"
        
        # If the test case has expected features, validate them
        if 'expected_features' in test_case:
            expected = test_case['expected_features']
            actual = result['features']
            
            # Test each feature that has expected values
            for feature_name in ['colors', 'text', 'fonts']:
                if feature_name in expected:
                    assert feature_name in actual, f"{feature_name} not found in response"
                    self._test_feature_match(expected[feature_name], actual[feature_name], feature_name)
        
        # Check that the status is either 'success' or 'partial'
        assert result['status'] in ['success', 'partial'], \
            f"Expected status 'success' or 'partial', got '{result['status']}'"
        
        # Validate metadata
        if 'metadata' in result:
            assert 'features_processed' in result['metadata'], "Missing 'features_processed' in metadata"
            assert 'features_failed' in result['metadata'], "Missing 'features_failed' in metadata"
            assert 'processing_time_ms' in result['metadata'], "Missing 'processing_time_ms' in metadata"
            
            assert result['metadata']['features_processed'] > 0, "No features were processed"
            
            if result['status'] == 'success':
                assert result['metadata']['features_failed'] == 0, \
                    "Status is 'success' but some features failed"
            elif result['status'] == 'partial':
                assert result['metadata']['features_failed'] > 0, \
                    "Status is 'partial' but no features failed"
    
    def test_analyze_image_invalid_file(self):
        """Test the analyze endpoint with an invalid file."""
        test_text_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'invalid.txt')
        
        with open(test_text_path, 'w') as f:
            f.write("This is not an image")
            
        with open(test_text_path, 'rb') as f:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("invalid.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Unsupported image format" in str(data["detail"]["error"]["message"])
    
    def test_analyze_image_no_file(self):
        """Test the analyze endpoint with no file."""
        response = client.post("/api/v1/analyze")
        
        assert response.status_code in [400, 422]  # Either validation error or missing file error
        data = response.json()
        assert "detail" in data


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

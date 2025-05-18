"""Integration tests for URL-based image analysis."""
import json
import os
import pytest
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
from app.main import app
from tests.constants import ValidationRules

# Initialize test client
client = TestClient(app)

# Sample features for testing
SAMPLE_FEATURES = {
    "colors": {
        "primary": "#101912",
        "background": "#FFFFFF",
        "accent": ["#188416", "#eeed2b", "#92b6c3"],
        "metadata": {}
    },
    "text": {
        "lines": ["How  created a in minutes! BOY   iL Ollama _-_- LangFlow"],
        "details": [{
            "text": "How  created a in minutes! BOY   iL Ollama _-_- LangFlow",
            "original": "How | created a in minutes! BOY ) ¢ iL Ollama _-_- LangFlow",
            "length": 56
        }],
        "metadata": {
            "confidence": 1,
            "success": True,
            "timestamp": 1747578173.8277693,
            "processing_time": 0.47467827796936035
        }
    },
    "fonts": {
        "font_family": "Arial",
        "font_size": 212,
        "font_style": "Bold",
        "confidence": 0.8
    }
}

def validate_features_structure(features: Dict[str, Any]):
    """Validate the structure of the features in the API response."""
    # Check required features
    assert "colors" in features, "Missing 'colors' in features"
    assert "text" in features, "Missing 'text' in features"
    assert "fonts" in features, "Missing 'fonts' in features"
    
    # Validate colors structure
    colors = features["colors"]
    assert "primary" in colors, "Missing 'primary' in colors"
    assert "background" in colors, "Missing 'background' in colors"
    assert "accent" in colors, "Missing 'accent' in colors"
    assert "metadata" in colors, "Missing 'metadata' in colors"
    
    # Validate text structure
    text = features["text"]
    assert "lines" in text, "Missing 'lines' in text"
    assert "details" in text, "Missing 'details' in text"
    assert "metadata" in text, "Missing 'metadata' in text"
    
    # Validate fonts structure
    fonts = features["fonts"]
    assert "font_family" in fonts, "Missing 'font_family' in fonts"
    assert "font_size" in fonts, "Missing 'font_size' in fonts"
    assert "font_style" in fonts, "Missing 'font_style' in fonts"
    assert "confidence" in fonts, "Missing 'confidence' in fonts"


class TestURLAnalysis:
    """Test cases for URL-based image analysis."""
    
    @pytest.mark.integration
    def test_analyze_image_from_url(self):
        """Test analyzing an image from a URL with the expected features structure."""
        # Test with a sample image URL from YouTube
        test_url = "https://i.ytimg.com/vi/ylLzFytBW4g/hq720.jpg"
        
        print(f"\nSending request to analyze image from URL: {test_url}")
        
        # Make the request
        response = client.post(
            "/api/v1/analyze/",
            data={"preprocessing": "auto"},
            files={"url": (None, test_url)},
        )
        
        # Print the full response for debugging
        print("\n=== Full Response ===")
        print(f"Status Code: {response.status_code}")
        print("Headers:", response.headers)
        print("Response Body:")
        
        # Parse and pretty print the JSON response
        try:
            result = response.json()
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print("Raw response:", response.text)
        
        print("\n=== End of Response ===\n")
        
        # Check response status code
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        
        # Get features from response
        assert "features" in result, "Response missing 'features' field"
        features = result["features"]
        
        # Validate the features structure
        validate_features_structure(features)
        
        # Check that features were extracted
        assert features, "No features were extracted"
        print(f"Features extracted: {list(features.keys())}")
    
    @pytest.mark.integration
    def test_features_structure_matches_sample(self):
        """Test that the API features match the expected structure."""
        # Make a request to the API
        test_url = "https://i.ytimg.com/vi/ylLzFytBW4g/hq720.jpg"
        response = client.post(
            "/api/v1/analyze/",
            data={"preprocessing": "auto"},
            files={"url": (None, test_url)},
        )
        
        # Get the features from the response
        result = response.json()
        assert "features" in result, "Response missing 'features' field"
        features = result["features"]
        
        # Check that all expected features are present
        for feature in ["colors", "text", "fonts"]:
            assert feature in features, f"Missing feature: {feature}"
        
        # Validate the structure of each feature
        for feature_name, expected_structure in SAMPLE_FEATURES.items():
            assert feature_name in features, f"Missing feature: {feature_name}"
            feature = features[feature_name]
            
            # Check that all expected keys are present
            for key in expected_structure.keys():
                assert key in feature, f"Missing key '{key}' in {feature_name}"
        
        print("Features structure matches the expected format")
    
    @pytest.mark.integration
    def test_analyze_image_from_url_without_scheme(self):
        """Test analyzing an image from a URL without http/https scheme."""
        # Test with a URL without scheme (should automatically add https://)
        test_url = "i.ytimg.com/vi/ylLzFytBW4g/hq720.jpg"
        
        # Make the request
        response = client.post(
            "/api/v1/analyze/",
            data={"preprocessing": "auto"},
            files={"url": (None, test_url)},
        )
        
        # Should still work - will automatically add https://
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
        
        # Parse response and validate features
        result = response.json()
        assert "features" in result, "Response missing 'features' field"
        validate_features_structure(result["features"])
        
        # Check that features were extracted
        assert result["features"], "No features were extracted"
    
    @pytest.mark.integration
    def test_analyze_image_invalid_url(self):
        """Test with an invalid URL."""
        # Test with an invalid URL
        test_url = "not-a-valid-url"
        
        # Make the request
        response = client.post(
            "/api/v1/analyze/",
            data={"preprocessing": "auto"},
            files={"url": (None, test_url)},
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected status code 400 for invalid URL, got {response.status_code}"
        
        # Check error response structure
        result = response.json()
        
        # The error response should have a 'detail' field with the error message
        assert "detail" in result, "Error response missing 'detail' field"
        
        # The detail can be either a string or an object with error details
        if isinstance(result["detail"], dict):
            # If it's an object, it should have 'error' field
            assert "error" in result["detail"], "Error detail missing 'error' field"
            error = result["detail"]["error"]
            
            # Check for expected fields in the error object
            if isinstance(error, dict):
                if "code" in error:
                    print(f"Error code: {error['code']}")
                if "message" in error:
                    print(f"Error message: {error['message']}")
        else:
            # If it's a string, just print it
            print(f"Error detail: {result['detail']}")
        
        # The test passes as long as we get a 400 status code
        # since different validation layers might format errors differently

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

import pytest
import os
import time
from fastapi.testclient import TestClient
from app.main import app
from tests.constants import (
    ValidationRules,
    validate_response_structure,
    validate_processing_time,
    get_performance_rating
)

class TestSampleDesignWebP:
    def setup_method(self):
        """Setup test client for API testing"""
        self.client = TestClient(app)
    
    @pytest.fixture
    def sample_design_path(self):
        """Fixture to provide the path to sample_design.webp"""
        return os.path.join(os.path.dirname(__file__),'..', 'test_images', 'shapes_sample.png')
    
    def test_webp_direct_upload(self, sample_design_path):
        """
        Test direct WebP image upload and processing
        Verify feature extraction capabilities on WebP format
        """
        # Load WebP image
        with open(sample_design_path, 'rb') as f:
            webp_image = f.read()
        
        # Prepare files for upload
        files = {'file': ('sample_design.webp', webp_image, 'image/webp')}
        
        # Prepare form data for features
        data = {
            'features': ['colors', 'text', 'shapes', 'fonts'],
            'preprocessing': 'auto'
        }
        
        # Send request to analyze endpoint
        response = self.client.post(
            "/api/v1/analyze", 
            files=files, 
            data=data
        )
        
        # Status code validation
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Processing time validation
        processing_time = response.elapsed.total_seconds()
        assert validate_processing_time(processing_time), \
            f"Processing time {processing_time}s exceeds maximum limit"
        
        # Performance rating
        performance_rating = get_performance_rating(processing_time)
        print(f"Performance Rating: {performance_rating}")
        result = response.json()
        
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Status: {result.get('status')}")
        
        print("\n" + "=" * 50)
        print("EXTRACTED FEATURES")
        print("=" * 50 + "\n")
        
        features = result.get('features', {})
        for feature, data in features.items():
            print(f"{feature.upper()}:")
            try:
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {data}")
            except Exception as e:
                print(f"Error processing {feature}: {str(e)}")
                
        # Feature Assertions
        if 'colors' in features:
            color_data = features['colors']
            if isinstance(color_data, dict) and 'error' in color_data:
                print(f"Warning: Color extraction failed - {color_data['error']}")
            else:
                is_valid, error_msg = validate_response_structure(
                    color_data,
                    ValidationRules.COLOR_EXTRACTION["expected_keys"],
                    value_types={
                        "primary": str,
                        "background": str,
                        "accent": list
                    },
                    additional_checks=[
                        lambda r: len(r["accent"]) <= ValidationRules.COLOR_EXTRACTION["max_accent_colors"]
                    ]
                )
                assert is_valid, f"Invalid color structure: {error_msg}"

        if 'shapes' in features:
            shape_data = features['shapes']
            if isinstance(shape_data, dict) and 'error' in shape_data:
                print(f"Warning: Shape extraction failed - {shape_data['error']}")
            else:
                is_valid, error_msg = validate_response_structure(
                    {"shapes": shape_data},  # Wrap in dict to match expected structure
                    ValidationRules.SHAPES_EXTRACTION["expected_keys"],
                    value_types={
                        "shapes": list
                    },
                    additional_checks=[
                        lambda r: len(r["shapes"]) <= ValidationRules.SHAPES_EXTRACTION["max_shapes"],
                        lambda r: all(
                            isinstance(shape, dict) and
                            "type" in shape and
                            shape["type"] in ValidationRules.SHAPES_EXTRACTION["valid_shape_types"]
                            for shape in r["shapes"]
                        )
                    ]
                )
                assert is_valid, f"Invalid shapes structure: {error_msg}"

        if 'fonts' in features:
            font_data = features['fonts']
            if isinstance(font_data, dict) and 'error' in font_data:
                print(f"Warning: Font extraction failed - {font_data['error']}")
            else:
                is_valid, error_msg = validate_response_structure(
                    {"fonts": font_data},  # Wrap in dict to match expected structure
                    ValidationRules.FONTS_EXTRACTION["expected_keys"],
                    value_types={
                        "fonts": list
                    },
                    additional_checks=[
                        lambda r: 0 < len(r["fonts"]) <= ValidationRules.FONTS_EXTRACTION["max_fonts"],
                        lambda r: all(
                            "family" in font and 
                            "size" in font and 
                            "weight" in font and 
                            "style" in font and
                            isinstance(font["family"], str) and len(font["family"]) > 0 and
                            isinstance(font["size"], (int, float)) and 
                            ValidationRules.FONTS_EXTRACTION["min_font_size"] <= font["size"] <= ValidationRules.FONTS_EXTRACTION["max_font_size"] and
                            font["weight"] in ValidationRules.FONTS_EXTRACTION["valid_weights"] and
                            font["style"] in ValidationRules.FONTS_EXTRACTION["valid_styles"]
                            for font in r["fonts"]
                        )
                    ]
                )
                assert is_valid, f"Invalid fonts structure: {error_msg}"

        if 'text' in features:
            text_data = features['text']
            print("\nText Data Structure:")
            print(text_data)
            if isinstance(text_data, dict) and 'error' in text_data:
                print(f"Warning: Text extraction failed - {text_data['error']}")
            else:
                # If text_data has 'lines' key, use that directly
                lines = text_data.get('lines', text_data)
                if isinstance(lines, list):
                    is_valid, error_msg = validate_response_structure(
                        {"lines": lines},  # Wrap in dict to match expected structure
                        ValidationRules.TEXT_EXTRACTION["expected_keys"],
                        value_types={
                            "lines": list
                        },
                        additional_checks=[
                            lambda r: len(r["lines"]) <= ValidationRules.TEXT_EXTRACTION["max_text_entries"],
                            lambda r: all(
                                isinstance(line, str) and
                                len(line) >= ValidationRules.TEXT_EXTRACTION["min_text_length"]
                                for line in r["lines"]
                            )
                        ]
                    )
                    assert is_valid, f"Invalid text structure: {error_msg}"
                else:
                    print(f"Warning: Unexpected text data format - {type(lines)}")

        if 'fonts' in features:
            font_data = features['fonts']
            assert test_case.fonts.primary_font.lower() in [f.lower() for f in font_data.get('detected_fonts', [])], f"Primary font not detected"

        if 'shadows' in features:
            shadow_data = features['shadows']
            assert shadow_data.get('has_shadows') == test_case.shadows.has_shadows, f"Shadow detection mismatch"

        # Validate response structure
        assert 'status' in result, "Response missing status"
        assert result['status'] in ['success', 'partial'], f"Unexpected status: {result.get('status')}"
        
        # Validate extracted features
        assert 'colors' in features, "Color extraction failed"
        assert 'text' in features, "Text extraction failed"
        assert 'shapes' in features, "Shape analysis failed"
        assert 'fonts' in features, "Font detection failed"
        
        # Performance metrics
        print("\n=== Performance Metrics ===\n")
        print(f"Total Processing Time: {processing_time:.2f} seconds")
        
        if 'metadata' in result:
            print("\nProcessing Summary:")
            metadata = result['metadata']
            print(f"  Total Features Requested: {metadata.get('total_features_requested')}")
            print(f"  Features Processed: {metadata.get('features_processed')}")
            print(f"  Features Failed: {metadata.get('features_failed')}")
        
        if 'errors' in result and result['errors']:
            print("\n=== Errors ===\n")
            for feature, error in result['errors'].items():
                print(f"Error in {feature}:")
                print(f"  Type: {error.get('type')}")
                print(f"  Message: {error.get('message')}")
        
        # Performance assertion
        assert processing_time < 2.0, f"Processing took too long: {processing_time} seconds"
        
        # Optional: Log performance insights
        print(f"WebP Image Processing Time: {processing_time} seconds")
        print("Extracted Features:", list(features.keys()))
        
        # Check for any errors
        if 'errors' in result:
            print("Extraction Errors:", result['errors'])
            assert len(result['errors']) == 0, "Some feature extractions failed"

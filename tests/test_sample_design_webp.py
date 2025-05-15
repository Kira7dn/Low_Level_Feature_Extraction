import pytest
import os
import time
from fastapi.testclient import TestClient
from app.main import app
from .test_cases import TestCase

class TestSampleDesignWebP:
    def setup_method(self):
        """Setup test client for API testing"""
        self.client = TestClient(app)
    
    @pytest.fixture
    def sample_design_path(self):
        """Fixture to provide the path to sample_design.webp"""
        return os.path.join(os.path.dirname(__file__), 'test_images', 'image.png')
    
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
        
        # Measure processing time
        start_time = time.time()
        
        # Send request to analyze endpoint
        response = self.client.post(
            "/analyze", 
            files=files, 
            data=data
        )
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get test case
        test_case = TEST_CASES["visa_card"]
        
        # Assertions
        assert response.status_code == 200, f"API request failed with status {response.status_code}"
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
            # Assert primary color
            assert color_data.get('primary') == test_case.colors.primary, f"Primary color mismatch. Expected {test_case.colors.primary}, got {color_data.get('primary')}"
            
            # Assert background color
            assert color_data.get('background') == test_case.colors.background, f"Background color mismatch. Expected {test_case.colors.background}, got {color_data.get('background')}"
            
            # Assert accent colors (order may vary)
            actual_accents = set(color_data.get('accent', []))
            expected_accents = set(test_case.colors.accent)
            assert actual_accents == expected_accents, f"Accent colors mismatch. Expected {expected_accents}, got {actual_accents}"

        if 'shapes' in features:
            shape_data = features['shapes']
            assert shape_data.get('total_shapes') == test_case.shapes.total_shapes, f"Total shapes mismatch"
            assert shape_data.get('rectangles') == test_case.shapes.rectangles, f"Rectangle count mismatch"

        if 'text' in features:
            text_data = features['text']
            for expected_text in test_case.text.content:
                assert any(expected_text.lower() in text.lower() for text in text_data.get('content', [])), f"Text '{expected_text}' not found"

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

import pytest
from typing import Any
import numpy as np
import cv2
import pytesseract
import time

from app.services.text_extractor import TextExtractor

class TestTextExtractor:
    @pytest.fixture
    def text_extractor(self):
        return TextExtractor()

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with text for testing"""
        # Create a white image
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        image.fill(255)
        
        # Add some text using cv2
        cv2.putText(
            image, 
            'Hello World', 
            (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 0), 
            2
        )
        
        return image

    def test_preprocess_image(self, text_extractor, sample_image):
        """Test image preprocessing"""
        preprocessed = text_extractor.preprocess_image(sample_image)
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2  # Expect 2D binary image
        assert preprocessed.dtype == np.uint8  # Expect binary image as uint8
        assert preprocessed.shape[0] == sample_image.shape[0]  # Height should match
        assert preprocessed.shape[1] == sample_image.shape[1]  # Width should match

    def test_extract_text_basic(self, text_extractor, sample_image):
        """Test basic text extraction"""
        result = text_extractor.extract_text(sample_image)
        
        # Check basic structure of the result
        assert 'lines' in result
        assert 'details' in result
        assert 'metadata' in result
        
        # Check that lines are extracted
        assert len(result['lines']) > 0
        assert len(result['details']) > 0
        
        # Verify the content of the extracted text
        assert 'Hello' in result['lines'][0] or 'World' in result['lines'][0]

    def test_confidence_threshold(self, sample_image):
        """Test confidence threshold filtering"""
        # Test with high confidence threshold to ensure no results
        high_threshold_result = TextExtractor.extract_text(
            sample_image, 
            confidence_threshold=0.99
        )
        
        # Test with low confidence threshold
        low_threshold_result = TextExtractor.extract_text(
            sample_image, 
            confidence_threshold=0.0
        )
        
        # High threshold should filter out all results
        assert len(high_threshold_result['lines']) == 0
        
        # Low threshold should keep results
        assert len(low_threshold_result['lines']) > 0

    def test_metadata_fields(self, text_extractor, sample_image):
        """Test metadata fields in extraction result"""
        result = text_extractor.extract_text(sample_image)
        
        # Check basic structure
        assert 'lines' in result
        assert 'details' in result
        assert 'metadata' in result
        
        # Check metadata fields
        metadata = result['metadata']
        assert 'total_lines' in metadata
        assert 'extracted_lines' in metadata
        assert 'confidence_threshold' in metadata
        
        # Validate metadata values
        assert metadata['total_lines'] > 0
        assert metadata['extracted_lines'] <= metadata['total_lines']
        assert metadata['confidence_threshold'] >= 0

    def test_postprocess_text(self, text_extractor):
        """Test text post-processing"""
        # Test various input scenarios
        test_cases = [
            "",  # Empty string
            "Hello, World! 123",  # Normal text
            "—Special© Characters—",  # Special characters
            "0 1 5 Numeric Confusion"  # Numeric substitution
        ]
        
        for text in test_cases:
            processed = text_extractor.postprocess_text(text)
            
            # Check result structure
            assert isinstance(processed, dict)
            assert 'lines' in processed
            assert 'details' in processed
            
            # Validate lines
            for line in processed['lines']:
                # Ensure no unwanted characters
                assert all(
                    char.isalnum() or char in ' .,!?:;\'"-' 
                    for char in line
                )

    def test_character_substitution(self, text_extractor):
        """Test character substitution in post-processing"""
        test_cases = [
            ('0', 'O'),  # Zero to O
            ('1', 'I'),  # One to I
            ('5', 'S'),  # Five to S
            ('—', '-'),  # Em dash to hyphen
            ('\'', "'"),  # Left single quote to apostrophe
            ('"', '"')   # Curly quotes to straight quotes
        ]
        
        for original, expected in test_cases:
            processed = text_extractor.postprocess_text(original)
            
            # Check that the expected substitution is in the processed lines
            assert any(
                expected in line 
                for line in processed['lines']
            ), f"Expected '{expected}' not found in processed text for input '{original}'"

    def test_performance_large_input(self, sample_image):
        """Test text processing performance with large input"""
        large_text = "A" * 10000  # Very large input
        
        import time
        start_time = time.time()
        processed = TextExtractor.postprocess_text(large_text)
        processing_time = time.time() - start_time
        
        # Ensure processing doesn't take too long (adjust threshold as needed)
        assert processing_time < 1.0, f"Processing took {processing_time} seconds, which is too slow"
        assert len(processed['lines']) > 0

    def test_edge_cases(self, sample_image):
        """Test various edge cases in text processing"""
        edge_cases = [
            None,           # None input
            "",             # Empty string
            "   ",          # Whitespace only
            "\n\t\r",       # Whitespace and control characters
            "🚀 Emoji 🌟",  # Emoji and special characters
            "123!@#$%^&*()" # Symbols and numbers
        ]

        for case in edge_cases:
            processed = TextExtractor.postprocess_text(case)
            
            # Validate basic structure
            assert isinstance(processed, dict)
            assert 'lines' in processed
            assert 'details' in processed

        # Ensure safe processing of various inputs
        assert isinstance(processed['lines'], list)
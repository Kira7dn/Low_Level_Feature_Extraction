import os
import time
import numpy as np
import cv2
import pytest
from app.services.text_extractor import TextExtractor
from app.services.image_processor import ImageProcessor
from tests.constants import validate_response_structure, validate_processing_time

@pytest.fixture
def text_extractor():
    return TextExtractor()

@pytest.fixture
def sample_text():
    return 'Hello World'

@pytest.fixture
def sample_image(sample_text):
    """Create a sample image with text for testing"""
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    cv2.putText(
        image, 
        sample_text, 
        (50, 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 0),  # Black text
        2
    )
    return image

@pytest.fixture
def empty_image():
    """Create an empty black image"""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def low_contrast_image():
    """Create a low contrast grayscale gradient image"""
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        image[:, i] = i
    return image

class TestTextExtractor:
    def test_initialization(self, text_extractor):
        """Test TextExtractor initialization"""
        assert text_extractor is not None
        assert isinstance(text_extractor, TextExtractor)

    def test_extract_from_file(self, text_extractor):
        """Test extracting text from a sample image file"""
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'test_images', 'text_sample.png')
        
        with open(test_image_path, 'rb') as f:
            cv_image = ImageProcessor.load_cv2_image(f.read())
        
        result = text_extractor.extract_text(cv_image)
        
        assert isinstance(result, dict)
        assert 'lines' in result
        assert isinstance(result['lines'], list)
        assert len(result['lines']) > 0
        assert all(isinstance(text, str) and text.strip() for text in result['lines'])

    def test_preprocessing(self, text_extractor, sample_image):
        """Test image preprocessing"""
        preprocessed = text_extractor.preprocess_image(sample_image)
        
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2  # 2D binary image
        assert preprocessed.dtype == np.uint8
        assert preprocessed.shape == sample_image.shape[:2]  # Same dimensions as input

    def test_basic_extraction(self, text_extractor, sample_image, sample_text):
        """Test basic text extraction from generated image"""
        start_time = time.time()
        result = text_extractor.extract_text(sample_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=['lines', 'details', 'metadata'],
            value_types={
                'lines': list,
                'details': list,
                'metadata': dict
            },
            nested_keys={
                'metadata': ['confidence', 'timestamp']
            },
            context='basic_extraction'
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context='basic_extraction'
        )
        assert is_valid, error_msg
        
        # Verify the content of the extracted text
        assert len(result['lines']) > 0
        assert any('Hello' in line or 'World' in line for line in result['lines'])

    def test_confidence_threshold(self, text_extractor, sample_image):
        """Test confidence threshold filtering"""
        # Test with high confidence threshold
        high_threshold_result = text_extractor.extract_text(
            sample_image, 
            confidence_threshold=0.99
        )
        
        # Test with low confidence threshold
        low_threshold_result = text_extractor.extract_text(
            sample_image, 
            confidence_threshold=0.0
        )
        
        # High threshold should filter out most/all results
        assert len(high_threshold_result['lines']) <= len(low_threshold_result['lines'])
        
        # Low threshold should keep more results
        assert len(low_threshold_result['lines']) > 0

    def test_metadata_fields(self, text_extractor, sample_image):
        """Test metadata fields in extraction result"""
        start_time = time.time()
        result = text_extractor.extract_text(sample_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure with focus on metadata
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=['metadata'],
            value_types={
                'metadata': dict
            },
            nested_keys={
                'metadata': ['confidence', 'timestamp']
            },
            additional_checks=[
                lambda r: 0 <= r['metadata']['confidence'] <= 1,  # Confidence range check
                lambda r: isinstance(r['metadata']['confidence'], (int, float))  # Type check
            ],
            context='metadata_validation'
        )
        assert is_valid, error_msg
        
        # Validate processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            context='metadata_validation'
        )
        assert is_valid, error_msg
        
        # Check metadata structure
        metadata = result['metadata']
        assert all(key in metadata for key in ['total_lines', 'extracted_lines', 'confidence_threshold'])
        
        # Validate metadata values
        assert metadata['total_lines'] >= 0
        assert metadata['extracted_lines'] <= metadata['total_lines']
        assert metadata['confidence_threshold'] >= 0

    def test_empty_image(self, text_extractor, empty_image):
        """Test text extraction on an empty image"""
        start_time = time.time()
        result = text_extractor.extract_text(empty_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=['lines', 'details', 'metadata'],
            value_types={
                'lines': list,
                'details': list,
                'metadata': dict
            },
            additional_checks=[
                lambda r: len(r['lines']) == 0,  # Should have no lines
                lambda r: r['metadata']['confidence'] == 0.0  # No confidence for empty image
            ],
            context='empty_image'
        )
        assert is_valid, error_msg
        
        # Empty images should process quickly
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            max_time=1.0,  # Should be quick for empty image
            recommended_time=0.5,
            context='empty_image'
        )
        assert is_valid, error_msg

    def test_low_contrast_image(self, text_extractor, low_contrast_image):
        """Test text extraction on a low contrast image"""
        start_time = time.time()
        result = text_extractor.extract_text(low_contrast_image)
        elapsed_time = time.time() - start_time
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=['lines', 'details', 'metadata'],
            value_types={
                'lines': list,
                'details': list,
                'metadata': dict
            },
            additional_checks=[
                # Low contrast should result in lower confidence
                lambda r: r['metadata']['confidence'] < 0.7
            ],
            context='low_contrast_image'
        )
        assert is_valid, error_msg
        
        # Low contrast might need more processing time
        is_valid, error_msg = validate_processing_time(
            elapsed_time,
            max_time=3.0,  # Allow more time for contrast enhancement
            recommended_time=1.5,
            context='low_contrast_image'
        )
        assert is_valid, error_msg

    def test_postprocess_text(self, text_extractor):
        """Test text post-processing"""
        test_cases = [
            "",  # Empty string
            "Hello, World! 123",  # Normal text
            "—Special© Characters—",  # Special characters
            "0 1 5 Numeric Confusion"  # Numeric substitution
        ]
        
        for i, text in enumerate(test_cases):
            start_time = time.time()
            processed = text_extractor.postprocess_text(text)
            elapsed_time = time.time() - start_time
            
            # Validate response structure
            is_valid, error_msg = validate_response_structure(
                processed,
                expected_keys=['lines', 'details', 'metadata'],
                value_types={
                    'lines': list,
                    'details': list,
                    'metadata': dict
                },
                additional_checks=[
                    # Empty string should result in empty lines
                    lambda r: (len(r['lines']) == 0) if text == "" else True,
                    # Normal text should be preserved
                    lambda r: any('Hello' in line for line in r['lines']) if 'Hello' in text else True,
                    # Special characters should be handled
                    lambda r: not any('©' in line for line in r['lines']) if '©' in text else True
                ],
                context=f'postprocess_case_{i}'
            )
            assert is_valid, error_msg
            
            # Post-processing should be quick
            is_valid, error_msg = validate_processing_time(
                elapsed_time,
                max_time=0.5,  # Post-processing should be fast
                recommended_time=0.1,
                context=f'postprocess_case_{i}'
            )
            assert is_valid, error_msg
            
            # Validate lines contain only allowed characters
            for line in processed['lines']:
                assert all(
                    char.isalnum() or char in ' .,!?:;\'"- ' 
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
            assert any(
                expected in line 
                for line in processed['lines']
            ), f"Expected '{expected}' not found for input '{original}'"

    def test_performance_large_input(self, text_extractor):
        """Test text processing performance"""
        large_text = "A" * 10000
        
        start_time = time.time()
        processed = text_extractor.postprocess_text(large_text)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0, f"Processing took {processing_time}s"
        assert len(processed['lines']) > 0

    def test_edge_cases(self, text_extractor):
        """Test various edge cases in text processing"""
        edge_cases = [
            None,           # None input
            "",            # Empty string
            "   ",         # Whitespace only
            "\n\t\r",      # Control characters
            "🚀 Emoji 🌟", # Emoji
            "123!@#$%^&*()" # Symbols and numbers
        ]

        for case in edge_cases:
            processed = text_extractor.postprocess_text(case)
            assert isinstance(processed, dict)
            assert all(key in processed for key in ['lines', 'details'])
            assert isinstance(processed['lines'], list)

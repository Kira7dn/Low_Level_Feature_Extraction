"""Test text extraction from various images with expected results."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import pytest

from app.services.text_extractor import TextExtractor
from tests.constants import ValidationRules, validate_response_structure

# Configure root logger to only show warnings and above
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "tests" / "test_images"
TEST_DATA_FILE = PROJECT_ROOT / "tests" / "test_data" / "test_text_images.json"
MIN_MATCH_PERCENTAGE = ValidationRules.TEXT_EXTRACTION.get("min_match_percentage", 60.0)

def load_test_cases() -> list[dict]:
    """Load and return test cases from JSON file.
    
    Returns:
        List of test cases with image paths and expected results.
    """
    if not TEST_DATA_FILE.exists():
        logger.warning("Test data file not found: %s", TEST_DATA_FILE)
        return []
    
    try:
        with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = data.get('test_cases', [])
        logger.info("Loaded %d test cases", len(test_cases))
        return test_cases
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Error loading test cases: %s", str(e))
        return []

# Load test cases once at module level
TEST_CASES = load_test_cases()


class TextExtractionTestBase:
    """Base class for text extraction tests with common functionality."""
    
    def setup_method(self) -> None:
        """Initialize test environment."""
        self.debug_dir = PROJECT_ROOT / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        
    def _get_full_image_path(self, image_path: str) -> Path:
        """Convert relative image path to absolute path."""
        path = Path(image_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    
    def _calculate_match_percentage(self, expected: str, extracted: str) -> float:
        """Calculate match percentage between expected and extracted text."""
        normalized_expected = " ".join(expected.split())
        normalized_extracted = " ".join(extracted.split())
        
        # Special handling for non-alphanumeric characters
        if any(not c.isalnum() and not c.isspace() for c in expected):
            expected_chars = {c.lower() for c in expected if not c.isspace()}
            extracted_chars = {c.lower() for c in extracted if not c.isspace()}
            
            if not expected_chars:
                return 0.0
                
            matching = expected_chars.intersection(extracted_chars)
            return len(matching) / len(expected_chars) * 100
            
        # Word-level matching for regular text
        expected_words = {w.lower() for w in normalized_expected.split() if len(w) > 1}
        if not expected_words:
            return 0.0
            
        extracted_words = {w.lower() for w in normalized_extracted.split() if len(w) > 1}
        matching = expected_words.intersection(extracted_words)
        return len(matching) / len(expected_words) * 100


class TestTextExtraction(TextExtractionTestBase):
    """Test cases for text extraction functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment."""
        super().setup_method()
        self.test_cases = TEST_CASES
    
    def _extract_text_from_image(self, image_path: str) -> tuple[dict, str]:
        """Extract text from image and validate the result.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (result_dict, extracted_text)
        """
        full_path = self._get_full_image_path(image_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Test image not found: {full_path}")
        
        image = cv2.imread(str(full_path))
        if image is None:
            raise IOError(f"Failed to read image: {full_path}")
        
        # Save original image for debugging
        debug_path = self.debug_dir / f"{full_path.stem}_debug.png"
        cv2.imwrite(str(debug_path), image)
        
        # Extract text
        result = TextExtractor.extract_text(image)
        
        # Validate response structure
        is_valid, error_msg = validate_response_structure(
            result,
            expected_keys=["lines", "details", "metadata"],
            value_types={
                "lines": list,
                "details": list,
                "metadata": dict
            },
            additional_checks=[
                lambda r: all(isinstance(line, str) for line in r["lines"]),
                lambda r: all(isinstance(detail, dict) and 
                           all(key in detail for key in ["text", "confidence", "bbox"]) 
                           for detail in r["details"])
            ]
        )
        
        if not is_valid:
            pytest.fail(f"Invalid response structure: {error_msg}")
            
        # Join all lines with spaces for text comparison
        full_text = " ".join(line.strip() for line in result['lines'] if line.strip())
        return result, full_text
    
    @pytest.mark.parametrize("test_case", TEST_CASES or [], 
                           ids=lambda x: x.get('description', 'unnamed') if TEST_CASES else 'no_test_cases')
    def test_text_extraction(self, test_case: dict, capsys):
        """Test text extraction from an image and verify against expected text."""
        if not test_case:
            pytest.skip("No test cases loaded")
        
        # Validate test case structure
        required_fields = {'image_path', 'expected_text', 'description'}
        if missing := required_fields - set(test_case):
            pytest.fail(f"Test case missing required fields: {', '.join(missing)}")
        
        try:
            # Extract text
            result, extracted_text = self._extract_text_from_image(test_case['image_path'])
            
            # Validate response structure
            assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
            assert 'metadata' in result, "Result missing 'metadata' key"
            assert result['metadata'].get('success', False), \
                f"Text extraction failed: {result.get('metadata', {}).get('error', 'Unknown error')}"
            
            # Calculate match percentage
            match_percentage = self._calculate_match_percentage(
                test_case['expected_text'], 
                extracted_text
            )
            
            # Output results
            with capsys.disabled():
                print(f"\nTest: {test_case['description']}")
                print(f"Expected: {test_case['expected_text']}")
                print(f"Result:  {extracted_text}")
                print(f"Match:   {match_percentage:.1f}%")
                print(f"Lines:   {result['lines']}")
                print(f"Details: {[d['text'] for d in result['details']]}")
                print(f"Confidence: {sum(d['confidence'] for d in result['details'])/len(result['details']):.1f}%")
                print("-" * 50)
            
            # Verify minimum match percentage
            assert match_percentage >= MIN_MATCH_PERCENTAGE, (
                f"Expected at least {MIN_MATCH_PERCENTAGE}% match. Got {match_percentage:.1f}%\n"
                f"Expected: {test_case['expected_text']}\n"
                f"Got: {extracted_text}"
            )
            
        except Exception as e:
            logger.error("Test failed with error: %s", str(e))
            raise

# This allows running the test directly with python -m pytest tests/unit/test_text_extraction_cases.py -v -s
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))

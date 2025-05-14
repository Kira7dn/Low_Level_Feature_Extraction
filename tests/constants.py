# Test Constants and Validation Utilities

import re
from typing import List, Union, Callable

# Performance Configuration
MAX_PROCESSING_TIME = 5.0  # seconds
RECOMMENDED_PROCESSING_TIME = 2.0  # seconds for optimal performance

# Common Validation Constants
class ValidationRules:
    """Centralized validation rules for different endpoint responses"""
    
    # Color Extraction Validation
    COLOR_EXTRACTION = {
        "expected_keys": ["primary", "background", "accent"],
        "hex_color_pattern": re.compile(r'^#[0-9A-Fa-f]{6}$'),
        "max_accent_colors": 5
    }
    
    # Text Extraction Validation
    TEXT_EXTRACTION = {
        "expected_keys": ["text"],
        "min_text_length": 1,
        "max_text_entries": 50
    }
    
    # Shapes Extraction Validation
    SHAPES_EXTRACTION = {
        "expected_keys": ["shapes"],
        "valid_shape_types": ["rectangle", "circle", "polygon", "triangle", "ellipse"],
        "max_shapes": 20,
        "min_coordinates": 2
    }
    
    # Shadows Extraction Validation
    SHADOWS_EXTRACTION = {
        "expected_keys": ["shadow_level"],
        "valid_levels": ["Low", "Moderate", "High"]
    }
    
    # Fonts Extraction Validation
    FONTS_EXTRACTION = {
        "expected_keys": ["fonts"],
        "valid_weights": ["normal", "bold", "light"],
        "valid_styles": ["normal", "italic", "oblique"],
        "max_fonts": 10,
        "min_font_size": 6,
        "max_font_size": 72
    }

def validate_hex_color(color: str) -> bool:
    """
    Validate a hex color code
    
    Args:
        color (str): Color to validate
    
    Returns:
        bool: True if valid hex color, False otherwise
    """
    return bool(ValidationRules.COLOR_EXTRACTION["hex_color_pattern"].match(color))

def validate_response_structure(
    result: dict, 
    expected_keys: List[str], 
    additional_checks: List[Callable[[dict], bool]] = None
) -> bool:
    """
    Validate response structure and keys
    
    Args:
        result (dict): Response to validate
        expected_keys (List[str]): Keys that must be present
        additional_checks (List[Callable], optional): Additional validation functions
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    # Check if all expected keys are present
    if not all(key in result for key in expected_keys):
        return False
    
    # Run additional checks if provided
    if additional_checks:
        return all(check(result) for check in additional_checks)
    
    return True

def validate_processing_time(
    elapsed_time: float, 
    max_time: float = MAX_PROCESSING_TIME
) -> bool:
    """
    Validate processing time of an API response
    
    Args:
        elapsed_time (float): Time taken to process the request
        max_time (float, optional): Maximum allowed processing time
    
    Returns:
        bool: True if processing time is within acceptable range
    """
    return elapsed_time < max_time

def get_performance_rating(elapsed_time: float) -> str:
    """
    Determine performance rating based on processing time
    
    Args:
        elapsed_time (float): Time taken to process the request
    
    Returns:
        str: Performance rating (Excellent, Good, Acceptable, Slow)
    """
    if elapsed_time < 1.0:
        return "Excellent"
    elif elapsed_time < 2.0:
        return "Good"
    elif elapsed_time < MAX_PROCESSING_TIME:
        return "Acceptable"
    else:
        return "Slow"

# Error Codes and Messages
class ErrorCodes:
    """Standardized error codes for API testing"""
    INVALID_FILE_FORMAT = 400
    NO_FILE_UPLOADED = 422
    PROCESSING_ERROR = 500
    VALIDATION_ERROR = 422

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Supported Image Formats
SUPPORTED_IMAGE_FORMATS = [
    "image/png", 
    "image/jpeg", 
    "image/bmp", 
    "image/webp"
]

# Maximum File Size
MAX_FILE_SIZE_MB = 5  # 5 megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

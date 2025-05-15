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
        "expected_keys": ["lines"],
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

def validate_hex_color(color: str) -> tuple[bool, str]:
    """
    Validate a hex color code
    
    Args:
        color (str): Color to validate
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(color, str):
        return False, f"Expected string, got {type(color)}"
    
    if not color:
        return False, "Color string is empty"
        
    if not color.startswith('#'):
        return False, "Color must start with '#'"
        
    if len(color) != 7:
        return False, f"Invalid color length: {len(color)}, expected 7 (#RRGGBB)"
        
    if not ValidationRules.COLOR_EXTRACTION["hex_color_pattern"].match(color):
        return False, f"Invalid hex color format: {color}, expected #RRGGBB"
        
    return True, ""

def validate_response_structure(
    result: dict, 
    expected_keys: List[str], 
    value_types: dict = None,
    nested_keys: dict = None,
    additional_checks: List[Callable[[dict], bool]] = None,
    context: str = None
) -> tuple[bool, str]:
    """
    Validate response structure, keys, and value types with detailed error reporting
    
    Args:
        result (dict): Response to validate
        expected_keys (List[str]): Keys that must be present
        value_types (dict, optional): Expected types for values {key: type}
        nested_keys (dict, optional): Expected nested keys {parent_key: [child_keys]}
        additional_checks (List[Callable], optional): Additional validation functions
        context (str, optional): Additional context for error messages
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    
    Example error messages:
        - "Missing required keys: key1, key2 in endpoint response"
        - "Invalid type for 'count': expected int, got str (value: '123')"
        - "Missing nested keys in metadata: timestamp, version"
    """
    context_prefix = f"[{context}] " if context else ""
    
    # Validate input type
    if not isinstance(result, dict):
        return False, f"{context_prefix}Expected dict, got {type(result).__name__}"

    # Check required keys
    missing_keys = [key for key in expected_keys if key not in result]
    if missing_keys:
        return False, f"{context_prefix}Missing required keys: {', '.join(missing_keys)}"
    
    # Validate value types
    if value_types:
        for key, expected_type in value_types.items():
            if key in result:
                value = result[key]
                if not isinstance(value, expected_type):
                    actual_type = type(value).__name__
                    value_preview = str(value)[:50] + '...' if len(str(value)) > 50 else str(value)
                    return False, (
                        f"{context_prefix}Invalid type for '{key}': "
                        f"expected {expected_type.__name__}, got {actual_type} "
                        f"(value: {value_preview})"
                    )
    
    # Validate nested structure
    if nested_keys:
        for parent_key, child_keys in nested_keys.items():
            if parent_key not in result:
                continue
            parent = result[parent_key]
            if not isinstance(parent, dict):
                return False, (
                    f"{context_prefix}Invalid type for nested key '{parent_key}': "
                    f"expected dict, got {type(parent).__name__}"
                )
            missing_nested = [key for key in child_keys if key not in parent]
            if missing_nested:
                return False, (
                    f"{context_prefix}Missing nested keys in '{parent_key}': "
                    f"{', '.join(missing_nested)}"
                )
    
    # Run additional validation checks
    if additional_checks:
        for check in additional_checks:
            try:
                if not check(result):
                    return False, f"{context_prefix}Validation failed: {check.__name__}"
            except Exception as e:
                return False, (
                    f"{context_prefix}Error in validation {check.__name__}: "
                    f"{str(e)}"
                )
    
    return True, ""

def validate_processing_time(
    elapsed_time: float, 
    max_time: float = MAX_PROCESSING_TIME,
    min_time: float = 0.0,
    recommended_time: float = RECOMMENDED_PROCESSING_TIME,
    context: str = None
) -> tuple[bool, str]:
    """
    Validate processing time of an API response and provide detailed feedback
    
    Args:
        elapsed_time (float): Time taken to process the request
        max_time (float): Maximum allowed processing time
        min_time (float): Minimum expected processing time (to catch suspicious fast responses)
        recommended_time (float): Recommended processing time threshold
        context (str, optional): Additional context for error messages
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    context_prefix = f"[{context}] " if context else ""
    
    # Validate input type
    if not isinstance(elapsed_time, (int, float)):
        return False, f"{context_prefix}Invalid time type: expected number, got {type(elapsed_time).__name__}"
    
    # Check for negative time
    if elapsed_time < 0:
        return False, f"{context_prefix}Invalid negative processing time: {elapsed_time}s"
    
    # Check for suspiciously fast responses
    if elapsed_time < min_time:
        return False, f"{context_prefix}Suspiciously fast processing time: {elapsed_time:.3f}s (min: {min_time:.3f}s)"
    
    # Check for exceeded max time
    if elapsed_time > max_time:
        return False, (
            f"{context_prefix}Processing time exceeded maximum: {elapsed_time:.3f}s "
            f"(max: {max_time:.3f}s, recommended: {recommended_time:.3f}s)"
        )
    
    # Warning for responses exceeding recommended time
    if elapsed_time > recommended_time:
        return True, (
            f"{context_prefix}Processing time above recommended threshold: {elapsed_time:.3f}s "
            f"(recommended: {recommended_time:.3f}s)"
        )
    
    return True, ""

def get_performance_rating(elapsed_time: float) -> tuple[str, float, str]:
    """
    Determine performance rating based on processing time
    
    Args:
        elapsed_time (float): Time taken to process the request
    
    Returns:
        tuple[str, float, str]: (rating, score, description)
            - rating: Performance rating (Excellent, Good, Acceptable, Slow)
            - score: Numeric score from 0-100
            - description: Detailed performance description
    """
    # Calculate performance score (0-100)
    if elapsed_time <= RECOMMENDED_PROCESSING_TIME * 0.5:
        score = 100
        rating = "Excellent"
        desc = "Response time well under recommended threshold"
    elif elapsed_time <= RECOMMENDED_PROCESSING_TIME:
        score = 80 + (RECOMMENDED_PROCESSING_TIME - elapsed_time) / (RECOMMENDED_PROCESSING_TIME * 0.5) * 20
        rating = "Good"
        desc = "Response time within recommended threshold"
    elif elapsed_time <= MAX_PROCESSING_TIME:
        score = 60 + (MAX_PROCESSING_TIME - elapsed_time) / (MAX_PROCESSING_TIME - RECOMMENDED_PROCESSING_TIME) * 20
        rating = "Acceptable"
        desc = "Response time within maximum threshold but above recommended"
    else:
        score = max(0, 60 - (elapsed_time - MAX_PROCESSING_TIME))
        rating = "Slow"
        desc = "Response time exceeds maximum threshold"
    
    return rating, round(score, 2), desc

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

import logging
from typing import Optional

import numpy as np
from fastapi import HTTPException, status

from app.services.font_detector import FontDetector
from app.services.models import FontFeatures

logger = logging.getLogger(__name__)

def extract_font_features(
    image: np.ndarray, request_id: str
) -> Optional[FontFeatures]:
    """Extract font features from the text in the image.
    
    This function analyzes the typography of text present in the image to identify
    font families, styles, and other typographic properties.
    
    Args:
        image: Input image as a numpy array in BGR format.
        request_id: Unique identifier for the request, used for logging.
        
    Returns:
        FontFeatures object containing the extracted font information or None if extraction fails.
        
    Raises:
        HTTPException: If there's an error during font detection.
    """
    try:
        logger.info(f"Starting font detection for request {request_id}")
        
        # Initialize the font detector
        detector = FontDetector()
        
        # Detect fonts (synchronous call)
        font_features = detector.detect_font(image)
        
        logger.info(f"Successfully detected fonts for request {request_id}")
        return font_features
        
    except Exception as e:
        logger.error(f"Error detecting fonts for request {request_id}: {str(e)}", exc_info=True)
        # Return None on error (font detection is optional)
        return None

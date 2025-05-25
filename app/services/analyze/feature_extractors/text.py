import logging
from typing import Optional

import numpy as np
from fastapi import HTTPException, status

from app.services.text_extractor import TextExtractor
from app.api.v1.models.analyze import TextFeatures

logger = logging.getLogger(__name__)

def extract_text_features(
    image: np.ndarray, request_id: str
) -> Optional[TextFeatures]:
    """Extract text features from the image using OCR.
    
    This function performs Optical Character Recognition (OCR) on the input image
    to extract text content and related metadata.
    
    Args:
        image: Input image as a numpy array in BGR format.
        request_id: Unique identifier for the request, used for logging.
        
    Returns:
        TextFeatures object containing the extracted text information or None if extraction fails.
        
    Raises:
        HTTPException: If there's an error during text extraction.
    """
    try:
        logger.info(f"Starting text extraction for request {request_id}")
        
        # Initialize the text extractor
        extractor = TextExtractor()
        
        # Extract text (synchronous call)
        text_features = extractor.extract_text(image)
        
        logger.info(f"Successfully extracted text for request {request_id}")
        return text_features
        
    except Exception as e:
        logger.error(f"Error extracting text for request {request_id}: {str(e)}", exc_info=True)
        # Return default TextFeatures on error
        return TextFeatures(
            lines=[],
            details=[],
            metadata={
                'success': False,
                'error': str(e),
                'timestamp': 0.0,
                'processing_time': 0.0
            }
        )

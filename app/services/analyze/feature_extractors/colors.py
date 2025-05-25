import logging
from typing import Optional

import numpy as np
from fastapi import HTTPException, status

from app.services.color_extractor import ColorExtractor
from app.api.v1.models.analyze import ColorFeatures

logger = logging.getLogger(__name__)


def extract_color_features(
    image: np.ndarray, request_id: str
) -> Optional[ColorFeatures]:
    """Extract color features from the image.
    
    This function analyzes the input image to identify and extract dominant colors,
    including the primary color, background color, and accent colors.
    
    Args:
        image: Input image as a numpy array in BGR format.
        request_id: Unique identifier for the request, used for logging.
            
    Returns:
        ColorFeatures object containing the extracted color information or None if extraction fails.
        
    Raises:
        HTTPException: If there's an error during color extraction.
    """
    try:
        logger.info(f"Starting color extraction for request {request_id}")
        
        # Initialize the color extractor
        extractor = ColorExtractor()
        
        # Extract colors (synchronous call)
        colors = extractor.extract_colors(image)
        
        logger.info(f"Successfully extracted colors for request {request_id}")
        return colors
        
    except Exception as e:
        logger.error(f"Error extracting colors for request {request_id}: {str(e)}", exc_info=True)
        # Return default ColorFeatures on error
        return ColorFeatures(
            primary='#000000',
            background='#FFFFFF',
            accent=['#666666', '#999999', '#CCCCCC'],
            metadata={
                'success': False,
                'error': str(e),
                'timestamp': 0.0,
                'processing_time': 0.0
            }
        )

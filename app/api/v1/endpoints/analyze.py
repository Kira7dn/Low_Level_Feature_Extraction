"""
Main router configuration for the analyze endpoints.

This module sets up the main FastAPI router and includes all the route modules.
"""

from fastapi import APIRouter, Query, HTTPException, status
from pydantic import HttpUrl
from typing import List, Optional
import time
import logging
from app.core.config import settings

# Import models for type hints and response model
from app.api.v1.models.analyze import UnifiedAnalysisResponse, FeatureType
from app.services.analyze.color_extractor import ColorExtractor
from app.services.analyze.font_detector import FontDetector
from app.services.analyze.image_processor import ImageProcessor
from app.services.analyze.text_extractor import TextExtractor
from app.utils.image_validator import validate_image

from app.services.analyze.utils import (
    download_image,
    validate_and_preprocess_image,
    process_feature_results,
)

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create the main router
router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)


# Register the route
@router.post(
    "/",
    response_model=UnifiedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an image",
    description="""
    Analyze an image from URL and extract visual features.
    
    This endpoint processes an image from the provided URL to extract various visual 
    features including colors, text content, and font information. The endpoint is 
    designed to handle a variety of image types and sizes with automatic preprocessing 
    to improve feature extraction accuracy.
    """,
    responses={
        status.HTTP_200_OK: {"description": "Analysis completed successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request parameters"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_image(
    image_url: HttpUrl = Query(..., description="URL of the image to process"),
) -> UnifiedAnalysisResponse:
    """
    Analyze an image from a URL and extract the requested features.

    Args:
        url: The URL of the image to analyze.
        features: List of features to extract. If None, all features will be extracted.
        preprocessing: Preprocessing mode to apply to the image.

    Returns:
        UnifiedAnalysisResponse containing the extracted features and metadata.
    """
    request_id = f"req-{int(time.time())}"
    logger.info(f"Starting image analysis request {request_id}")
    start_time = time.time()

    try:
        # Download and validate the image
        logger.info(f"Processing image from URL: {image_url}")
        image_bytes, content_type = await download_image(str(image_url))

        # Preprocess the image
        image = await validate_and_preprocess_image(
            image_bytes,
            request_id,
            preprocessing="auto",
        )
        # Determine which features to extract
        # if features is None:
        features = list(FeatureType)

        # Run feature extraction synchronously
        feature_results = []
        for feature in features:
            try:
                if feature == FeatureType.COLORS:
                    result = ColorExtractor.extract_colors(image)
                elif feature == FeatureType.TEXT:
                    result = TextExtractor.extract_text(image)
                elif feature == FeatureType.FONTS:
                    result = FontDetector.detect_font(image)
                else:
                    continue
                feature_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {feature}: {str(e)}")
                feature_results.append(e)

        # Process the results
        processing_time_ms = (time.time() - start_time) * 1000
        response = process_feature_results(
            features, feature_results, processing_time_ms, request_id
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )

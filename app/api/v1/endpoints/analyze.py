"""
Main router configuration for the analyze endpoints.

This module sets up the main FastAPI router and includes all the route modules.
"""

from fastapi import APIRouter, Form, HTTPException, status
from pydantic import HttpUrl
from typing import List, Optional

# Import models for type hints and response model
from app.api.v1.models.analyze import UnifiedAnalysisResponse, FeatureType

# Import the main analyze function from the analyzer module
from app.services.analyze.analyzer import analyze_image


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
async def analyze_image_endpoint(
    url: HttpUrl = Form(..., description="URL of the image to process"),
    features: Optional[List[FeatureType]] = Form(None, description="List of features to extract"),
    preprocessing: str = Form("auto", description="Preprocessing mode to apply to the image")
):
    """
    Analyze an image from URL and extract visual features.
    
    Args:
        url: URL of the image to analyze. Must be a valid image URL pointing to a
            supported image format (PNG, JPG, JPEG, WEBP). Maximum file size is 10MB.
        features: List of features to extract. If not provided, all features will be extracted.
        preprocessing: Preprocessing mode to apply to the image. Options: 'none', 'auto', 'high_quality', 'performance'.
            
    Returns:
        UnifiedAnalysisResponse: A structured response containing the analysis results.
        
    Raises:
        HTTPException: With appropriate status code for different error scenarios.
    """
    try:
        return await analyze_image(url, features, preprocessing)
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        # Log the error and return a 500 response
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error in analyze_image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request",
        )

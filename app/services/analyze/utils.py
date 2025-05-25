import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import cv2
import numpy as np
from fastapi import HTTPException, status
from pydantic import HttpUrl

from app.api.v1.models.analyze import (
    UnifiedAnalysisResponse,
    AnalysisMetadata,
    FeatureError,
    FeatureType,
)

logger = logging.getLogger(__name__)


class PreprocessingMode(str, Enum):
    """Enum for image preprocessing modes."""

    NONE = "none"
    AUTO = "auto"
    HIGH_QUALITY = "high_quality"
    PERFORMANCE = "performance"


async def download_image(
    url: str,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    timeout: int = 30,
) -> tuple[bytes, str]:
    """
    Download an image from a URL with size and timeout limits.

    Args:
        url: The URL of the image to download.
        max_size: Maximum allowed image size in bytes.
        timeout: Request timeout in seconds.

    Returns:
        A tuple containing (image_data, content_type).

    Raises:
        HTTPException: If download fails or validation fails.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download image. Status: {response.status}",
                    )

                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"URL does not point to a valid image. Content-Type: {content_type}",
                    )

                # Read the image data in chunks to respect max_size
                image_data = bytearray()
                async for chunk in response.content.iter_chunked(8192):  # 8KB chunks
                    image_data.extend(chunk)
                    if len(image_data) > max_size:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Image size exceeds maximum allowed size of {max_size} bytes",
                        )

                return bytes(image_data), content_type

    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image: {str(e)}",
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Image download timed out after {timeout} seconds",
        )


async def validate_and_preprocess_image(
    image_bytes: bytes,
    request_id: str,
    preprocessing: str,
) -> np.ndarray:
    """
    Validate and preprocess the input image.

    Args:
        image_bytes: Raw image data.
        preprocessing: Preprocessing mode.
        request_id: Request ID for logging.

    Returns:
        Preprocessed image as a numpy array.
    """
    try:
        # Convert bytes to numpy array
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode image. The file may be corrupted or in an unsupported format.",
            )

        # Apply preprocessing based on the mode
        if preprocessing == "none":
            pass  # No preprocessing
        elif preprocessing == "auto":
            # Auto preprocessing - resize large images for better performance
            max_dim = 2000
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        elif preprocessing == "high_quality":
            # High quality preprocessing - maintain aspect ratio but limit size
            max_dim = 4000
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        elif preprocessing == "performance":
            # Performance mode - aggressively resize for fastest processing
            max_dim = 1000
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        return image

    except Exception as e:
        logger.error(f"Error in validate_and_preprocess_image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation or preprocessing failed: {str(e)}",
        )


def process_feature_results(
    features_requested: List[FeatureType],
    results: List[Union[Dict[str, Any], Exception]],
    processing_time_ms: float,
    request_id: str,
) -> UnifiedAnalysisResponse:
    """
    Process the results from feature extraction and format the response.

    Args:
        features_requested: List of requested feature types.
        results: List of results from feature extraction.
        processing_time_ms: Total processing time in milliseconds.
        request_id: Request ID for logging.

    Returns:
        Formatted UnifiedAnalysisResponse.
    """
    features = {}
    errors = {}

    # Process each result
    for feature, result in zip(features_requested, results):
        feature_name = feature.value

        if isinstance(result, Exception):
            # Handle errors
            errors[feature_name] = FeatureError(
                code=type(result).__name__,
                message=str(result),
                severity="error" if not isinstance(result, Warning) else "warning",
            )
        else:
            # Add successful result
            features[feature_name] = result

    # Determine overall status
    if not errors:
        status = "success"
    elif len(errors) == len(features_requested):
        status = "failure"
    else:
        status = "partial"

    # Create metadata
    metadata = AnalysisMetadata(
        total_features_requested=len(features_requested),
        features_processed=len(features),
        features_failed=len(errors),
        processing_time_ms=processing_time_ms,
    )

    # Create and return the response
    return UnifiedAnalysisResponse(
        status=status,
        request_id=request_id,
        features=features,
        errors=errors,
        metadata=metadata,
    )

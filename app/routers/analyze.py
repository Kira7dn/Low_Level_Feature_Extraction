from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import aiohttp
import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, HttpUrl

# Local imports
from ..core.config import settings
from ..services.color_extractor import ColorExtractor, ColorPalette
from ..services.font_detector import FontDetector
from ..services.image_processor import ImageProcessor
from ..services.models import FontFeatures, TextFeatures
from ..services.text_extractor import TextExtractor
from ..utils.image_validator import validate_image

# Configure logger
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)

# ===== Response Models =====

class FeatureError(BaseModel):
    """Error details for a failed feature extraction.
    
    Attributes:
        code: A unique identifier for the error type in SCREAMING_SNAKE_CASE.
        message: Human-readable explanation of the error.
        severity: Indicates error severity level.
            - 'error': Critical issue that prevents feature extraction.
            - 'warning': Non-critical issue that allows partial processing.
    """
    code: str = Field(..., description="Error code in SCREAMING_SNAKE_CASE")
    message: str = Field(..., description="Human-readable error message")
    severity: Literal["error", "warning"] = Field(
        "error",
        description="Error severity level ('error' or 'warning')",
    )

class AnalysisMetadata(BaseModel):
    """Metadata about the analysis process.
    
    Attributes:
        total_features_requested: Total number of features requested for extraction.
            Must be >= 0.
        features_processed: Number of features successfully processed.
            Must be >= 0 and <= total_features_requested.
        features_failed: Number of features that failed to process.
            Must be >= 0 and <= total_features_requested.
        processing_time_ms: Total processing time in milliseconds.
            Includes all feature extraction and preprocessing time.
    """
    total_features_requested: int = Field(
        ..., 
        description="Total number of features requested for extraction",
        ge=0
    )
    features_processed: int = Field(
        ..., 
        description="Number of features successfully processed",
        ge=0
    )
    features_failed: int = Field(
        ..., 
        description="Number of features that failed to process",
        ge=0
    )
    processing_time_ms: float = Field(
        ..., 
        description="Total processing time in milliseconds",
        ge=0.0
    )

class UnifiedAnalysisResponse(BaseModel):
    """Response model for unified image analysis.
    
    This model represents the standardized response format for all image analysis requests,
    containing the extracted features, any errors that occurred, and processing metadata.
    
    Attributes:
        status: Indicates the overall status of the analysis.
            - 'success': All requested features were extracted successfully.
            - 'partial': Some features were extracted, but some failed.
            - 'failure': No features could be extracted.
        request_id: Unique identifier for the request, useful for debugging.
        features: Dictionary of extracted features where keys are feature names
            (e.g., 'colors', 'text') and values are the corresponding feature objects.
        errors: Dictionary of errors that occurred during processing, keyed by feature name.
            Empty if no errors occurred.
        metadata: Additional metadata about the analysis process.
    """
    status: Literal["success", "partial", "failure"]
    request_id: str
    features: Dict[
        str,
        Optional[Union[Dict[str, Any], ColorPalette, TextFeatures, FontFeatures]],
    ] = Field(..., description="Extracted features keyed by feature name")
    errors: Dict[str, FeatureError] = Field(
        default_factory=dict,
        description="Errors that occurred during processing, keyed by feature name",
    )
    metadata: AnalysisMetadata = Field(..., description="Metadata about the analysis process")

    class Config:
        json_encoders = {
            "ColorPalette": lambda v: v.dict() if hasattr(v, "dict") else v,
            "TextFeatures": lambda v: v.dict() if hasattr(v, "dict") else v,
            "FontFeatures": lambda v: v.dict() if hasattr(v, "dict") else v,
        }
        schema_extra = {
            "example": {
                "status": "success",
                "request_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
                "features": {
                    "colors": {
                        "primary": "#FF5733",
                        "background": "#FFFFFF",
                        "accent": ["#33FF57", "#3357FF"],
                    },
                    "text": {
                        "lines": ["Sample text"],
                        "details": [],
                        "metadata": {},
                    },
                },
                "errors": {},
                "metadata": {
                    "total_features_requested": 2,
                    "features_processed": 2,
                    "features_failed": 0,
                    "processing_time_ms": 123.45,
                },
            }
        }

# ===== Helper Functions =====

async def download_image(
    url: Union[str, HttpUrl],
    max_size: int = 10 * 1024 * 1024,
    timeout: int = 30,
) -> tuple[bytes, str]:
    """Download an image from a URL with size and timeout limits.
    
    Args:
        url: The URL of the image to download. Can be a string or HttpUrl.
            If string doesn't have a scheme, 'https://' will be added.
        max_size: Maximum allowed image size in bytes (default: 10MB).
        timeout: Request timeout in seconds (default: 30).
        
    Returns:
        A tuple containing (image_data, content_type).
        
    Raises:
        HTTPException: If download fails or validation fails.
        ValueError: If URL format is invalid.
    """
    timeout = aiohttp.ClientTimeout(total=timeout)
    
    # Convert to string and clean up the URL
    url_str = str(url).strip()
    
    # Add https:// if no scheme is present
    if not url_str.startswith(('http://', 'https://')):
        url_str = f'https://{url_str}'
        logger.debug(f"Added https:// scheme to URL: {url_str}")
    
    # Basic URL validation
    try:
        # Simple validation - check if it looks like a URL
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url_str):
            raise ValueError("Invalid URL format")
            
        # Try to parse with urllib for basic validation
        from urllib.parse import urlparse
        parsed = urlparse(url_str)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Missing scheme or netloc")
            
    except Exception as e:
        logger.error(f"Invalid URL format: {url_str} - {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "INVALID_URL",
                    "message": f"Invalid URL format: {str(e)}"
                }
            }
        )
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url_str) as response:
                if response.status != 200:
                    error_msg = f"Failed to download image: HTTP {response.status}"
                    logger.error(f"{error_msg} from {url_str}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": {
                                "code": "DOWNLOAD_FAILED",
                                "message": error_msg
                            }
                        }
                    )
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"URL does not point to a valid image. Content type: {content_type}",
                    )
                
                # Stream the download with size limit
                content_length = int(response.headers.get('content-length', 0))
                if content_length > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image size exceeds maximum allowed size of {max_size} bytes"
                    )
                
                chunks = []
                size = 0
                async for chunk in response.content.iter_chunked(8192):
                    size += len(chunk)
                    if size > max_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Image size exceeds maximum allowed size of {max_size} bytes"
                        )
                    chunks.append(chunk)
                
                return b''.join(chunks), content_type
                
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {str(e)}"
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Image download timed out"
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while downloading the image"
        )

# ===== Router and Endpoints =====

class PreprocessingMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    HIGH_QUALITY = "high_quality"
    PERFORMANCE = "performance"

class ImageSource(BaseModel):
    """Model for image source (either URL or file upload)."""
    url: Optional[HttpUrl] = Field(
        None,
        description="URL of the image to process. Either 'url' or 'file' must be provided.",
        example="https://example.com/image.jpg"
    )
    file: Optional[UploadFile] = Field(
        None,
        description="Image file to upload. Either 'url' or 'file' must be provided."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/image.jpg"
            }
        }

class FeatureType(str, Enum):
    COLORS = "colors"
    TEXT = "text"
    FONTS = "fonts"

@router.post(
    "/",
    response_model=UnifiedAnalysisResponse,
    status_code=status.HTTP_200_OK,
    response_description="Comprehensive analysis of image features",
    responses={
        status.HTTP_200_OK: {"model": UnifiedAnalysisResponse},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid URL or image format"},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "Image file too large"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Unsupported image format"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
    summary="Analyze image from URL",
    description="Process an image from a URL and extract features like colors, text, and fonts.",
)
async def analyze_image(
    url: HttpUrl = Form(..., description="URL of the image to process")
) -> UnifiedAnalysisResponse:
    """Analyze an image from URL and extract visual features.
    
    This endpoint processes an image from the provided URL to extract various visual 
    features including colors, text content, and font information. The endpoint is 
    designed to handle a variety of image types and sizes with automatic preprocessing 
    to improve feature extraction accuracy.
    
    Args:
        url: URL of the image to analyze. Must be a valid image URL pointing to a
            supported image format (PNG, JPG, JPEG, WEBP). Maximum file size is 10MB.
            
    Returns:
        UnifiedAnalysisResponse: A structured response containing:
            - status: Overall status of the analysis
            - request_id: Unique identifier for the request
            - features: Dictionary of extracted features
            - errors: Any errors that occurred during processing
            - metadata: Processing metrics and timing information
    
    Raises:
        HTTPException: With appropriate status code for different error scenarios:
            - 400 (Bad Request): Invalid URL or image format
            - 413 (Payload Too Large): Image file exceeds 10MB limit
            - 415 (Unsupported Media Type): Unsupported image format
            - 500 (Internal Server Error): Unexpected server error
    
    Example Request:
        ```bash
        curl -X POST "http://localhost:8000/analyze/" \
             -F "url=https://example.com/image.jpg"
        ```
    """
    request_id = str(uuid4())
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Starting image analysis request {request_id}")

    try:
        logger.info(f"Processing image from URL: {url}")
        
        # Download the image
        try:
            image_bytes, content_type = await download_image(
                url,
                max_size=10 * 1024 * 1024,  # 10MB
                timeout=30
            )
            logger.info(f"Downloaded {len(image_bytes)} bytes, content-type: {content_type}")
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Failed to process URL: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_URL",
                        "message": f"Invalid URL or failed to download image: {str(e)}"
                    }
                }
            )
        
        # Validate image content type
        if not content_type or not content_type.startswith('image/'):
            logger.error(f"Invalid content type: {content_type}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_IMAGE_TYPE",
                        "message": f"Unsupported image format: {content_type}"
                    }
                }
            )
        
        # Process the image with default settings
        cv_image = await validate_and_preprocess_image(
            image_bytes, 
            PreprocessingMode.AUTO,
            request_id
        )
        
        # Log image details after preprocessing
        if cv_image is not None:
            logger.info(f"Processed image - shape: {cv_image.shape}, dtype: {cv_image.dtype}")
        else:
            logger.error("Failed to process image - cv_image is None")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": "IMAGE_PROCESSING_ERROR",
                        "message": "Failed to process the image"
                    }
                }
            )
        
        # Define all available features and their extraction functions
        feature_map = {
            FeatureType.COLORS: lambda: extract_color_features(cv_image, request_id),
            FeatureType.TEXT: lambda: extract_text_features(cv_image, request_id),
            FeatureType.FONTS: lambda: extract_font_features(cv_image, request_id)
        }
        
        # Extract all features
        features_to_extract = list(feature_map.keys())
        
        # Execute tasks in parallel
        tasks = [
            asyncio.create_task(feature_map[feature]()) 
            for feature in features_to_extract
            if feature in feature_map
        ]
        
        feature_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate processing time
        processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Process results
        return process_feature_results(
            features_to_extract,
            feature_results,
            processing_time_ms,
            request_id
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in request {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "request_id": request_id,
                "type": type(e).__name__
            }
        )

async def validate_and_preprocess_image(
    image_bytes: bytes,
    preprocessing: PreprocessingMode,
    request_id: str
) -> np.ndarray:
    """Validate and preprocess the input image.
    
    Args:
        image_bytes: Raw image data
        preprocessing: Preprocessing mode
        request_id: Request ID for logging
        
    Returns:
        np.ndarray: Preprocessed image in BGR format
    """
    try:
        # Apply standard preprocessing
        cv_image = ImageProcessor.auto_process_image(
            image_bytes,
            max_width=settings.IMAGE_MAX_WIDTH,
            max_height=settings.IMAGE_MAX_HEIGHT,
            quality=settings.IMAGE_QUALITY
        )
        
        # Apply additional preprocessing based on mode
        if preprocessing == PreprocessingMode.HIGH_QUALITY:
            # Example: Apply denoising
            cv_image = cv2.fastNlMeansDenoisingColored(
                cv_image,
                None,
                h=3,
                hColor=3,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
        logger.info(
            f"Processed image - shape: {cv_image.shape}, "
            f"dtype: {cv_image.dtype}, "
            f"min: {cv_image.min()}, "
            f"max: {cv_image.max()}"
        )
        return cv_image
        
    except Exception as e:
        logger.error(f"Image validation failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image file",
                "request_id": request_id,
                "type": type(e).__name__
            }
        )
        
def process_feature_results(
    features_requested: list[FeatureType],
    results: list[Union[dict[str, Any], Exception]],
    processing_time_ms: float,
    request_id: str,
) -> UnifiedAnalysisResponse:
    """Process and format the results from feature extraction.
    
    This function takes the raw results from feature extraction functions and
    transforms them into a standardized response format. It handles both successful
    results and errors, and computes the overall status of the analysis.
    
    Args:
        features_requested: List of feature types that were requested for extraction.
        results: List of results corresponding to each requested feature. Each item
            can be either a dictionary/object with the extracted features or an Exception
            if the feature extraction failed.
        processing_time_ms: Total time taken for processing all features, in milliseconds.
        request_id: Unique identifier for the request, used for logging and tracing.
            
    Returns:
        A structured response containing the processed results, any errors that occurred,
        and processing metadata.
            
    Raises:
        ValueError: If the number of results doesn't match the number of requested features.
            
    Notes:
        - The function calculates the overall status based on the success/failure of
          individual feature extractions.
        - Each feature result is validated and converted to the appropriate Pydantic model.
        - Errors are captured and included in the response without failing the entire request.
    """
    if len(features_requested) != len(results):
        error_msg = (
            f"Mismatch between requested features ({len(features_requested)}) "
            f"and results ({len(results)})"
        )
        logger.error(f"[{request_id}] {error_msg}")
        raise ValueError(error_msg)
    
    features: dict[str, Any] = {}
    errors: dict[str, FeatureError] = {}
    
    for feature, result in zip(features_requested, results, strict=True):
        feature_name = feature.value
        
        if isinstance(result, Exception):
            # Handle error case
            error_msg = str(result)
            error_type = type(result).__name__
            logger.error(
                f"[{request_id}] Error processing {feature_name}: {error_msg}",
                exc_info=result,
            )
            
            errors[feature_name] = FeatureError(
                code=error_type,
                message=error_msg,
                severity="error",
            )
        else:
            # Handle successful result
            try:
                # Convert result to appropriate model based on feature type
                if feature == FeatureType.COLORS and isinstance(result, dict):
                    features[feature_name] = ColorPalette(**result)
                elif feature == FeatureType.TEXT and isinstance(result, dict):
                    features[feature_name] = TextFeatures(**result)
                elif feature == FeatureType.FONTS and isinstance(result, dict):
                    features[feature_name] = FontFeatures(**result)
                elif isinstance(result, dict):
                    # For any other dictionary results, include as-is
                    features[feature_name] = result
                else:
                    # For non-dict results, include as-is
                    features[feature_name] = result
                    
                logger.debug(f"[{request_id}] Successfully processed {feature_name} feature")
                
            except Exception as e:
                error_msg = f"Failed to process {feature_name} result: {str(e)}"
                logger.error(f"[{request_id}] {error_msg}", exc_info=e)
                errors[feature_name] = FeatureError(
                    code="PROCESSING_ERROR",
                    message=error_msg,
                    severity="error",
                )
    
    # Calculate overall status
    total_features = len(features_requested)
    failed_features = len(errors)
    successful_features = len(features)
    
    if failed_features == 0:
        status = "success"
    elif successful_features == 0:
        status = "failure"
    else:
        status = "partial"
    
    # Create metadata
    metadata = AnalysisMetadata(
        total_features_requested=total_features,
        features_processed=successful_features,
        features_failed=failed_features,
        processing_time_ms=round(processing_time_ms, 2),  # Round to 2 decimal places
    )
    
    logger.info(
        f"[{request_id}] Processed {successful_features}/{total_features} features "
        f"(failed: {failed_features}) in {processing_time_ms:.2f}ms"
    )
    
    # Ensure features is always a dictionary, never None
    if not features:
        features = {}
        
    return UnifiedAnalysisResponse(
        status=status,
        request_id=request_id,
        features=features,
        errors=errors or {},
        metadata=metadata,
    )

async def extract_color_features(image: np.ndarray, request_id: str) -> ColorPalette:
    """Extract color features from the image.
    
    This function analyzes the input image to identify and extract dominant colors,
    including the primary color, background color, and accent colors. The extraction
    is optimized for both speed and accuracy, using clustering algorithms to identify
    the most representative colors in the image.
    
    Args:
        image: Input image as a numpy array in BGR format. Should be preprocessed
            and validated before being passed to this function.
        request_id: Unique identifier for the request, used for logging.
            
    Returns:
        An object containing the extracted color information:
            - primary: The most dominant color in the image (hex code)
            - background: The detected background color (hex code)
            - accent: List of up to 5 accent colors (hex codes)
            
    Raises:
        ValueError: If the input image is invalid or cannot be processed.
        RuntimeError: If color extraction fails due to an internal error.
        
    Example:
        ```python
        palette = await extract_color_features(cv2_image, "req-123")
        print(f"Primary color: {palette.primary}")
        print(f"Background: {palette.background}")
        print(f"Accent colors: {', '.join(palette.accent)}")
        ```
    """
    try:
        logger.info(f"Starting color extraction for request {request_id}")
        
        # Get colors from the extractor
        colors = ColorExtractor.extract_colors(image)
        logger.info(f"Extracted colors: {colors}")
        
        # Create ColorPalette instance
        result = ColorPalette(
            primary=colors.primary,
            background=colors.background,
            accent=colors.accent or []
        )
        
        logger.info(f"Formatted color features: {result.dict()}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting colors for request {request_id}: {str(e)}", exc_info=True)
        # Return default ColorPalette on error
        return ColorPalette()

async def extract_text_features(image: np.ndarray, request_id: str) -> TextFeatures:
    """Extract text features from the image using OCR.
    
    This function performs Optical Character Recognition (OCR) on the input image
    to extract text content and related metadata. It supports multiple languages
    and can handle various text orientations and layouts.
    
    Args:
        image: Input image as a numpy array in BGR format. Should be preprocessed
            to enhance text visibility if necessary.
        request_id: Unique identifier for the request, used for logging.
        
    Returns:
        An object containing:
            - lines: List of extracted text strings, one per line
            - details: List of text detection details including bounding boxes
            - metadata: Additional information about the extraction process
            
    Raises:
        ValueError: If the input image is invalid or contains no text.
        RuntimeError: If the OCR engine fails to process the image.
        
    Notes:
        - The function automatically handles text orientation detection.
        - Confidence scores are included in the metadata for each detected text element.
        - The function attempts to maintain the original text structure and formatting.
        
    Example:
        ```python
        text_features = await extract_text_features(cv2_image, "req-123")
        for i, line in enumerate(text_features.lines, 1):
            print(f"Line {i}: {line}")
        ```
    """
    try:
        logger.info(f"Starting text extraction for request {request_id}")
        
        # Extract text using the TextExtractor
        text_result = TextExtractor.extract_text(image)
        logger.info(f"Extracted text: {text_result}")
        
        # Ensure we have the expected structure
        if not hasattr(text_result, 'lines') or not isinstance(text_result.lines, list):
            text_result.lines = []
            
        if not hasattr(text_result, 'details') or not isinstance(text_result.details, list):
            text_result.details = []
            
        if not hasattr(text_result, 'metadata') or not isinstance(text_result.metadata, dict):
            text_result.metadata = {}
            
        # Ensure metadata has required fields
        if 'confidence' not in text_result.metadata:
            text_result.metadata['confidence'] = 100.0 if text_result.lines else 0.0
            
        if 'success' not in text_result.metadata:
            text_result.metadata['success'] = bool(text_result.lines)
            
        if 'timestamp' not in text_result.metadata:
            from time import time
            text_result.metadata['timestamp'] = time()
            
        if 'processing_time' not in text_result.metadata:
            text_result.metadata['processing_time'] = 0.0
        
        # Create TextFeatures instance from the result
        result = TextFeatures(
            lines=text_result.lines,
            details=text_result.details,
            metadata=text_result.metadata
        )
        
        logger.info(f"Formatted text features: {result.dict()}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting text for request {request_id}: {str(e)}", exc_info=True)
        # Return empty TextFeatures with default values on error
        from time import time
        return TextFeatures(
            lines=[],
            details=[],
            metadata={
                'confidence': 0.0,
                'success': False,
                'timestamp': time(),
                'processing_time': 0.0,
                'error': str(e)
            }
        )

async def extract_font_features(image: np.ndarray, request_id: str) -> Optional[FontFeatures]:
    """Extract font features from the text in the image.
    
    This function analyzes the typography of text present in the image to identify
    font families, styles, and other typographic properties. It works best with
    clear, high-contrast text and may return None if the text is not clearly visible.
    
    Args:
        image: Input image as a numpy array in BGR format. Should contain
            clearly visible text for accurate font detection.
        request_id: Unique identifier for the request, used for logging.
        
    Returns:
        An object containing font information if detection is successful, or None
        if no text is detected or if the fonts cannot be reliably identified.
        The object includes:
            - font_family: Detected font family name
            - font_style: Font style (e.g., 'regular', 'bold', 'italic')
            - font_size: Approximate font size in points
            - confidence: Detection confidence score (0-1)
            
    Raises:
        ValueError: If the input image is invalid or contains no detectable text.
        
    Notes:
        - Font detection accuracy depends on text clarity and size.
        - The function may return None for very small or low-quality text.
        - Multiple font families in the same image may reduce detection accuracy.
        
    Example:
        ```python
        font_features = await extract_font_features(cv2_image, "req-123")
        if font_features:
            print(f"Font: {font_features.font_family} ({font_features.font_style})")
            print(f"Size: {font_features.font_size}pt")
            print(f"Confidence: {font_features.confidence:.1%}")
        ```
    """
    try:
        # Get font info from the detector (synchronous call)
        font_features = FontDetector.detect_font(image)
        
        if font_features is None:
            logger.warning(f"No font detected in image for request {request_id}")
            return FontFeatures(
                font_family='sans-serif',
                font_size=12.0,
                font_style='normal',
                confidence=0.0
            )
            
        return font_features
        
    except Exception as e:
        logger.error(f"Error extracting fonts for request {request_id}: {str(e)}")
        # Return None to indicate failure
        return None
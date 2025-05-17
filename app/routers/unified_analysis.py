from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
import cv2
import logging
from uuid import uuid4

# Import existing service layers
from ..services.image_processor import ImageProcessor
from ..services.color_extractor import ColorExtractor
from ..services.text_extractor import TextExtractor
from ..services.font_detector import FontDetector

# Import utility modules
from ..utils.image_validator import validate_image

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

class PreprocessingMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    HIGH_QUALITY = "high_quality"
    PERFORMANCE = "performance"

class FeatureType(str, Enum):
    COLORS = "colors"
    TEXT = "text"
    FONTS = "fonts"

@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Unified image analysis",
    response_description="Comprehensive analysis of image features"
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    preprocessing: PreprocessingMode = Form(
        PreprocessingMode.AUTO,
        description="Image preprocessing mode"
    ),
    features: Optional[List[FeatureType]] = Form(
        None,
        description="List of features to extract. If not specified, all features will be extracted."
    )
) -> Dict[str, Any]:
    """
    Analyze an image and extract specified or all available features.
    
    This endpoint processes an image to extract various features including colors, text, and fonts.
    Features are processed in parallel for better performance.
    
    Args:
        file (UploadFile): The image file to analyze. Supported formats: PNG, JPG, JPEG, WEBP.
        preprocessing (PreprocessingMode, optional): Image preprocessing mode. Defaults to 'auto'.
            - 'none': No preprocessing
            - 'auto': Automatic preprocessing based on image content
            - 'high_quality': Enhanced preprocessing for better accuracy (slower)
            - 'performance': Optimized for speed (may reduce accuracy)
        features (List[FeatureType], optional): List of features to extract. If not specified,
            all available features will be extracted. Available features:
            - 'colors': Extract dominant colors and color scheme
            - 'text': Extract text content and metadata
            - 'fonts': Identify font properties
    
    Returns:
        Dict[str, Any]: A structured response with the following schema:
        {
            'status': str,  # 'success', 'partial', or 'failure'
            'request_id': str,  # Unique request identifier
            'features': {
                'colors': {
                    'primary': str,  # Hex color code (e.g., '#007BFF')
                    'background': str,  # Hex color code
                    'accent': List[str]  # List of hex color codes
                },
                'text': {
                    'lines': List[str]  # List of extracted text lines
                },
                'fonts': {
                    'font_family': str  # Detected font family
                }
            },
            'errors': Dict[str, Dict[str, str]],  # Error information by feature
            'metadata': {
                'total_features_requested': int,
                'features_processed': int,
                'features_failed': int,
                'processing_time_ms': float
            }
        }
    
    Raises:
        HTTPException: If there's an error processing the request
            - 400: Invalid image format or parameters
            - 413: Image file too large
            - 415: Unsupported media type
            - 422: Validation error
            - 500: Internal server error
    """
    request_id = str(uuid4())
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Starting image analysis request {request_id}")
    
    try:
        # Log file upload details
        logger.info(f"File upload details - filename: {file.filename}, content_type: {file.content_type}")
        
        # Read file content synchronously
        image_bytes = file.file.read()
        logger.info(f"Read {len(image_bytes)} bytes from uploaded file")
        
        # Log first few bytes to verify file signature
        if image_bytes:
            logger.info(f"File signature (first 16 bytes): {image_bytes[:16].hex(' ')}")
        else:
            logger.warning("Uploaded file is empty")
            
        cv_image = await validate_and_preprocess_image(
            image_bytes, 
            preprocessing,
            request_id
        )
        
        # Log image details after preprocessing
        if cv_image is not None:
            logger.info(f"Processed image - shape: {cv_image.shape}, dtype: {cv_image.dtype}")
        else:
            logger.error("Failed to process image - cv_image is None")
            
        # Define all available features and their extraction functions
        feature_map = {
            FeatureType.COLORS: lambda: extract_color_features(cv_image, request_id),
            FeatureType.TEXT: lambda: extract_text_features(cv_image, request_id),
            FeatureType.FONTS: lambda: extract_font_features(cv_image, request_id)
        }
            
        # Determine which features to extract
        features_to_extract = features or list(feature_map.keys())
        
        # Execute selected tasks in parallel
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
) -> Any:
    """Validate and preprocess the input image."""
    try:
        # Validate image
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        
        # Ensure image is BGR (most services expect this)
        if len(cv_image.shape) == 2 or (len(cv_image.shape) == 3 and cv_image.shape[2] == 1):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            
        # Apply preprocessing based on mode
        if preprocessing != PreprocessingMode.NONE:
            max_dimension = {
                PreprocessingMode.AUTO: 2000,
                PreprocessingMode.HIGH_QUALITY: 4000,
                PreprocessingMode.PERFORMANCE: 1000
            }.get(preprocessing, 2000)
                
            height, width = cv_image.shape[:2]
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                    
                cv_image = cv2.resize(
                    cv_image, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_AREA
                )
                logger.debug(f"Resized image to {new_width}x{new_height} for request {request_id}")
                
        return cv_image
        
    except Exception as e:
        logger.error(f"Image validation failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid image file",
                "request_id": request_id,
                "type": type(e).__name__
            }
        )

def process_feature_results(
    features_requested: List[FeatureType],
    results: List[Any],
    processing_time_ms: float,
    request_id: str
) -> Dict[str, Any]:
    """Process and format the results from feature extraction."""
    response = {
        'status': 'success',
        'request_id': request_id,
        'features': {
            'colors': None,
            'text': None,
            'fonts': None
        },
        'errors': {},
        'metadata': {
            'total_features_requested': len(features_requested),
            'features_processed': 0,
            'features_failed': 0,
            'processing_time_ms': round(processing_time_ms, 2)
        }
    }
    
    for feature, result in zip(features_requested, results):
        if isinstance(result, Exception):
            response['status'] = 'partial' if response['status'] != 'failure' else 'failure'
            response['errors'][feature.value] = {
                'code': type(result).__name__,
                'message': str(result),
                'severity': 'error'
            }
            response['metadata']['features_failed'] += 1
        else:
            if feature == FeatureType.COLORS:
                response['features']['colors'] = {
                    'primary': result.get('primary'),
                    'background': result.get('background'),
                    'accent': result.get('accent')
                }
            elif feature == FeatureType.TEXT:
                response['features']['text'] = {
                    'lines': result.get('lines')
                }
            elif feature == FeatureType.FONTS:
                response['features']['fonts'] = {
                    'font_family': result.get('font_family')
                }
            response['metadata']['features_processed'] += 1
    
    return response

async def extract_color_features(image: Any, request_id: str) -> Dict[str, Any]:
    """Extract color features from the image.
    
    Returns:
        Dict: {
            'primary': str,       # Primary color in hex (e.g., '#007BFF')
            'background': str,    # Background color in hex
            'accent': List[str]   # List of accent colors in hex
        }
    """
    try:
        logger.info(f"Starting color extraction for request {request_id}")
        
        # Get colors from the extractor
        colors = ColorExtractor.extract_colors(image)
        logger.info(f"Extracted colors: {colors}")
        
        # Transform to match our schema
        result = {
            'primary': colors.get('primary', '#000000'),
            'background': colors.get('background', '#FFFFFF'),
            'accent': colors.get('accent', [])
        }
        
        logger.info(f"Formatted color features: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting colors for request {request_id}: {str(e)}", exc_info=True)
        # Return default values on error
        return {
            'primary': None,
            'background': None,
            'accent': None
        }

async def extract_text_features(image: Any, request_id: str) -> Dict[str, Any]:
    """Extract text features from the image.
    
    Returns:
        Dict: {
            'lines': List[str]  # List of extracted text lines
        }
    """
    try:
        # Get text from the extractor (synchronous call)
        text_result = TextExtractor.extract_text(image)
        
        # Transform to match our schema
        return {
            'lines': text_result.get('lines', [])
        }
    except Exception as e:
        logger.error(f"Error extracting text for request {request_id}: {str(e)}")
        raise

async def extract_font_features(image: Any, request_id: str) -> Dict[str, Any]:
    """Extract font features from the image.
    
    Returns:
        Dict: {
            'font_family': str  # Detected font family
        }
    """
    try:
        # Get font info from the detector (synchronous call)
        font_info = FontDetector.detect_font(image)  # Fixed method name from detect_fonts to detect_font
        
        if font_info is None:
            logger.warning(f"No font detected in image for request {request_id}")
            return {'font_family': 'sans-serif'}
            
        # Transform to match our schema
        return {
            'font_family': font_info.get('font_family', 'sans-serif')
        }
    except Exception as e:
        logger.error(f"Error extracting fonts for request {request_id}: {str(e)}")
        raise
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Dict, List, Optional
import asyncio
from app.utils.cache import cache_result

# Import existing service layers
from ..services.image_processor import ImageProcessor
from ..services.color_extractor import ColorExtractor
from ..services.text_extractor import TextExtractor
from ..services.shape_analyzer import ShapeAnalyzer
from ..services.font_detector import FontDetector
from ..services.shadow_analyzer import ShadowAnalyzer

# Import utility modules
from ..utils.image_validator import validate_image

# Import logging
import logging

# Configure logger
logger = logging.getLogger(__name__)


router = APIRouter()

@router.post(
    "/analyze",
    response_model=Dict[str, object],
    status_code=status.HTTP_200_OK,
    summary="Unified image analysis",
    response_description="Comprehensive analysis of image features"
)
@cache_result(ttl=600)  # Cache results for 10 minutes
async def analyze_image(
    file: UploadFile = File(...),
    preprocessing: Optional[str] = Form("auto"),
    features: Optional[List[str]] = Form(None)
) -> Dict[str, object]:
    """
    Analyze an image and extract specified or all available features.
    
    Parameters:
    - file: Image file to analyze
    - preprocessing: Image preprocessing mode (none, auto, high_quality, performance)
    - features: Optional list of features to extract. Available features:
      * 'colors': Color palette extraction
      * 'text': Text recognition
      * 'shapes': Shape and border analysis
      * 'fonts': Font detection
      * 'shadows': Shadow detection
      If not specified, all features will be extracted.
    
    Returns:
    A comprehensive JSON response with the following structure:
    {
        'status': 'success' | 'partial' | 'failure',
        'features': {
            '<feature_name>': <feature_result> | None
        },
        'errors': {
            '<failed_feature>': {
                'type': <error_type>,
                'message': <error_description>,
                'severity': 'critical' | 'warning'
            }
        },
        'metadata': {
            'total_features_requested': <int>,
            'features_processed': <int>,
            'features_failed': <int>
        }
    }
    
    - 'status' indicates overall request processing status
    - 'features' contains results for successful extractions
    - 'errors' provides details for failed feature extractions
    - 'metadata' offers processing summary information
    """
    try:
        # Validate and load image
        image_bytes = await validate_image(file)
        
        # Apply preprocessing based on selected mode
        # --- Unified Preprocessing Step ---
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        # Ensure image is BGR (most services expect this)
        if len(cv_image.shape) == 2 or (len(cv_image.shape) == 3 and cv_image.shape[2] == 1):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        # Resize if needed for performance
        max_dimension = 2000
        height, width = cv_image.shape[:2]
        if preprocessing == "performance" or (preprocessing == "auto" and (height > max_dimension or width > max_dimension)):
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # Optionally, add more preprocessing if needed (e.g., normalization)
        # --- End Unified Preprocessing Step ---
        
        # Define all available features and their extraction functions
        feature_map = {
            'colors': lambda: extract_color_features(cv_image),
            'text': lambda: extract_text_features(cv_image),
            'shapes': lambda: extract_shape_features(cv_image),
            'fonts': lambda: extract_font_features(cv_image),
            'shadows': lambda: extract_shadow_features(cv_image)
        }
        
        # Determine which features to extract
        if features is None:
            # Default to all features if no selection is made
            features = list(feature_map.keys())
        else:
            # Validate requested features
            invalid_features = set(features) - set(feature_map.keys())
            if invalid_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid feature(s) requested: {', '.join(invalid_features)}. Available features: {', '.join(feature_map.keys())}")
        
        # Prepare tasks for parallel execution based on selected features
        tasks = [
            asyncio.create_task(feature_map[feature]()) 
            for feature in features
        ]
        
        # Execute selected tasks in parallel
        feature_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        results = {
            'features': {},
            'errors': {},
            'status': 'success',
            'metadata': {
                'total_features_requested': len(features),
                'features_processed': 0,
                'features_failed': 0
            }
        }
        
        for feature, result in zip(features, feature_results):
            if isinstance(result, Exception):
                # Handle extraction failure
                results['features'][feature] = None
                results['errors'][feature] = {
                    'type': type(result).__name__,
                    'message': str(result),
                    'severity': 'critical'
                }
                results['metadata']['features_failed'] += 1
                results['status'] = 'partial' if results['status'] == 'success' else 'failure'
            else:
                # Successful extraction
                results['features'][feature] = result
                results['metadata']['features_processed'] += 1
        
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing image: {str(e)}"
        )

# Feature extraction helper functions
async def extract_color_features(image):
    """Extract color features from the image"""
    try:
        logger.info("Starting color feature extraction")
        img = ColorExtractor.preprocess_image(image) if hasattr(ColorExtractor, 'preprocess_image') else image
        result = ColorExtractor.extract_colors(img)
        logger.info("Color feature extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Color feature extraction failed: {str(e)}", exc_info=True)
        return {
            "error": "Color feature extraction failed",
            "details": str(e),
            "type": type(e).__name__
        }

async def extract_text_features(image):
    """Extract text features from the image"""
    try:
        logger.info("Starting text feature extraction")
        img = TextExtractor.preprocess_image(image) if hasattr(TextExtractor, 'preprocess_image') else image
        result = TextExtractor.extract_text(img)
        logger.info("Text feature extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Text feature extraction failed: {str(e)}", exc_info=True)
        return {
            "error": "Text feature extraction failed",
            "details": str(e),
            "type": type(e).__name__
        }

async def extract_shape_features(image):
    """Extract shape features from the image"""
    try:
        logger.info("Starting shape feature extraction")
        img = ShapeAnalyzer.preprocess_image(image) if hasattr(ShapeAnalyzer, 'preprocess_image') else image
        result = ShapeAnalyzer.detect_border_radius(img)
        logger.info("Shape feature extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Shape feature extraction failed: {str(e)}", exc_info=True)
        return {
            "error": "Shape feature extraction failed",
            "details": str(e),
            "type": type(e).__name__
        }

async def extract_font_features(image):
    """Extract font features from the image"""
    try:
        logger.info("Starting font feature extraction")
        img = FontDetector.preprocess_image(image) if hasattr(FontDetector, 'preprocess_image') else image
        result = FontDetector.detect_font(img)
        logger.info("Font feature extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Font feature extraction failed: {str(e)}", exc_info=True)
        return {
            "error": "Font feature extraction failed",
            "details": str(e),
            "type": type(e).__name__
        }

async def extract_shadow_features(image):
    """Extract shadow features from the image"""
    try:
        logger.info("Starting shadow feature extraction")
        img = ShadowAnalyzer.preprocess_image(image) if hasattr(ShadowAnalyzer, 'preprocess_image') else image
        result = ShadowAnalyzer.analyze_shadow_level(img)
        logger.info("Shadow feature extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Shadow feature extraction failed: {str(e)}", exc_info=True)
        return {
            "error": "Shadow feature extraction failed",
            "details": str(e),
            "type": type(e).__name__
        }

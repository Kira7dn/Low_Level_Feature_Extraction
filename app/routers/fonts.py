from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io

from ..services.font_detector import FontDetector
from ..services.image_processor import ImageProcessor
from ..utils.image_validator import validate_image

router = APIRouter(prefix="/fonts")

@router.post(
    "/extract-fonts", 
    response_model=dict, 
    status_code=status.HTTP_200_OK,
    summary="Extract font properties from an image",
    description="Detect font family, size, and weight from an uploaded design image"
)
async def extract_fonts(file: UploadFile = File(...)):
    """
    Endpoint to extract font properties from an uploaded image.
    
    Args:
        file (UploadFile): Uploaded image file
    
    Returns:
        dict: Font detection results with font family, size, and weight
    
    Raises:
        HTTPException: For invalid image or processing errors
    """
    try:
        # Validate and load image
        image_bytes = await validate_image(file)
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        
        # Detect font properties
        font_data = FontDetector.detect_font(cv_image)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK, 
            content=font_data
        )
    
    except ValueError as ve:
        # Handle specific validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(ve)
        )
    
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error detecting fonts: {str(e)}"
        )

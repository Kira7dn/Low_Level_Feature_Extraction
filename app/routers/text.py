from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Union, Optional

from app.services.text_extractor import TextExtractor
from app.services.models import TextFeatures
from app.utils.image_validator import validate_image
from app.services.image_processor import ImageProcessor

router = APIRouter(prefix="/text", tags=["Text Extraction"])

class TextExtractionResponse(TextFeatures):
    """
    Response model for text extraction from images
    
    Provides a list of text lines and metadata about the extraction process.
    """
    pass

class Base64TextRequest(BaseModel):
    """
    Request model for base64 encoded image text extraction
    """
    base64_image: str

@router.post("/extract", 
    response_model=TextExtractionResponse,
    responses={
        200: {
            "description": "Successful text extraction",
            "content": {
                "application/json": {
                    "example": {
                        "lines": ["Sample text line 1", "Sample text line 2"],
                        "details": [
                            {
                                "text": "Sample text line 1",
                                "confidence": 95.5,
                                "bbox": [10, 20, 100, 30]
                            }
                        ],
                        "metadata": {
                            "confidence": 95.5,
                            "success": True,
                            "timestamp": 1620000000.0,
                            "processing_time": 0.5
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid image file"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error during text extraction"}
    },
    summary="Extract Text from Image",
    description="""Detect and extract visible text from an uploaded image.
    
    Features:
    - Supports multiple languages
    - Provides confidence scores for extracted text
    - Returns bounding box locations for each text line
    - Includes detailed metadata about the extraction process
    
    Supported image formats: PNG, JPEG, BMP, WEBP
    """)
async def extract_text(
    file: UploadFile = File(..., description="Image file to analyze. Must be PNG, JPEG, or BMP."),
    lang: str = Query('eng', description="Language for OCR (default: English)"),
    config: str = Query('--oem 3 --psm 6', description="Tesseract configuration options")
) -> TextExtractionResponse:
    """
    Extract text from an uploaded image file
    
    Args:
        file: Image file to analyze
        lang: Language for OCR (default: English)
        config: Tesseract configuration options
    
    Returns:
        Comprehensive text extraction results
    """
    try:
        # Validate and load image
        print("Validating image...")
        image_bytes = await validate_image(file)
        print("Image validated successfully")
        try:
            cv_image = ImageProcessor.load_cv2_image(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
        
        # Extract text
        text_data = TextExtractor.extract_text(cv_image, lang=lang, config=config)
        print("text_data: ", text_data)
        
        # If text_data is already a TextFeatures object, return it directly
        if hasattr(text_data, 'dict'):
            return text_data
            
        # Otherwise, create a new TextExtractionResponse
        return TextExtractionResponse(**text_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        if "format not supported" in str(e).lower():
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": {
                        "message": "Unsupported image format",
                        "code": "INVALID_FILE_FORMAT"
                    }
                }
            )
        print("Error extracting text: ", str(e))
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

@router.post("/extract-base64", response_model=TextExtractionResponse, 
    summary="Extract Text from Base64 Image",
    description="""Detect and extract visible text from a base64 encoded image.
    
    Features:
    - Supports multiple languages
    - Provides confidence scores for extracted text
    - Returns bounding box locations for each text line
    
    Supported image formats: PNG, JPEG, BMP
    """,
    responses={
        200: {"description": "Successful text extraction"},
        400: {"description": "Invalid base64 image"},
        500: {"description": "Internal server error during text extraction"}
    })
async def extract_text_base64(
    request: Base64TextRequest,
    lang: str = Query('eng', description="Language for OCR (default: English)"),
    config: str = Query('--oem 3 --psm 6', description="Tesseract configuration options")
) -> TextExtractionResponse:
    """
    Extract text from a base64 encoded image
    
    Args:
        request: Base64 image request object
        lang: Language for OCR (default: English)
        config: Tesseract configuration options
    
    Returns:
        Comprehensive text extraction results
    """
    try:
        # Validate and load base64 image
        cv_image = ImageProcessor.load_cv2_image_from_base64(request.base64_image)
        
        # Extract text
        text_data = TextExtractor.extract_text(cv_image, lang=lang, config=config)
        
        # If text_data is already a TextFeatures object, return it directly
        if hasattr(text_data, 'dict'):
            return text_data
            
        # Otherwise, create a new TextExtractionResponse
        return TextExtractionResponse(**text_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

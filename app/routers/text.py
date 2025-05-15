from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Union

from app.services.text_extractor import TextExtractor
from app.utils.image_validator import validate_image
from app.services.image_processor import ImageProcessor

router = APIRouter(prefix="/text", tags=["Text Extraction"])

class TextExtractionResponse(BaseModel):
    """
    Response model for text extraction from images
    
    Provides a comprehensive breakdown of text extracted from an image,
    including the text lines, confidence scores, and bounding box locations.
    
    Attributes:
        lines (List[str]): Extracted text lines
        details (List[Dict]): Detailed information about each extracted text line
    """
    lines: List[str]
    details: List[Dict[str, Union[str, float, Dict[str, int]]]]

class Base64TextRequest(BaseModel):
    """
    Request model for base64 encoded image text extraction
    """
    base64_image: str

@router.post("/extract", response_model=TextExtractionResponse, 
    summary="Extract Text from Image",
    description="""Detect and extract visible text from an uploaded image.
    
    Features:
    - Supports multiple languages
    - Provides confidence scores for extracted text
    - Returns bounding box locations for each text line
    
    Supported image formats: PNG, JPEG, BMP
    """,
    responses={
        200: {"description": "Successful text extraction"},
        400: {"description": "Invalid image file"},
        500: {"description": "Internal server error during text extraction"}
    })
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
        image_bytes = await validate_image(file)
        try:
            cv_image = ImageProcessor.load_cv2_image(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
        
        # Extract text
        text_data = TextExtractor.extract_text(cv_image, lang=lang, config=config)
        
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
        
        return TextExtractionResponse(**text_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

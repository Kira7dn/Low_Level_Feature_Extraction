import base64
import io

from typing import List, Dict, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query
from pydantic import BaseModel
from PIL import Image
import numpy as np

from app.services.color_extractor import ColorExtractor
from app.utils.image_validator import validate_image
from app.utils.error_handler import ValidationException

router = APIRouter(prefix="/colors", tags=["Colors"])

class ColorExtractionResponse(BaseModel):
    """Response model for color extraction
    
    Provides a comprehensive breakdown of colors extracted from an image.
    
    Attributes:
        primary (str): The most dominant color in HEX format
        background (str): The background color in HEX format
        accent (List[str]): List of accent colors in HEX format
    
    Example:
        {
            "primary": "#FF0000",
            "background": "#0000FF",
            "accent": ["#00FF00", "#FFFF00"]
        }
    """
    primary: str
    background: str
    accent: List[str]

@router.post("/extract", response_model=ColorExtractionResponse, 
    summary="Extract Color Palette from Image",
    description="""Analyze an uploaded image to extract its dominant color palette.
    
    Features:
    - Identify up to 10 dominant colors
    - Return RGB, HEX, and human-readable color names
    - Determine primary and background colors
    
    Supported image formats: PNG, JPEG, BMP
    """,
    responses={
        200: {"description": "Successful color extraction"},
        400: {"description": "Invalid image file"},
        500: {"description": "Internal server error during color analysis"}
    })
async def extract_colors(
    file: UploadFile = File(..., description="Image file to analyze. Must be PNG, JPEG, or BMP."),
    n_colors: int = Query(5, ge=1, le=10, description="Number of dominant colors to extract (1-10). Default is 5.")
) -> ColorExtractionResponse:
    """Extract dominant colors from an uploaded image
    
    Analyzes the input image and returns a detailed color palette
    with RGB, HEX, and named colors.
    
    Args:
        file: Image file to analyze
        n_colors: Number of colors to extract (1-10)
    
    Returns:
        Comprehensive color palette details
    """
    # Validate the image
    try:
        validated_file = await validate_image(file)
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Open image with Pillow
    try:
        image = Image.open(io.BytesIO(validated_file))
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            # Create a white background image
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the background using alpha channel
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process the image: {str(e)}")
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Extract color palette
    try:
        color_analysis = ColorExtractor.analyze_palette(image_array, n_colors)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error analyzing color palette")
    
    return {
        'primary': color_analysis['primary'],
        'background': color_analysis['background'],
        'accent': color_analysis['accent']
    }

class Base64ImageRequest(BaseModel):
    base64_image: str

@router.post("/extract-base64", response_model=ColorExtractionResponse, 
    summary="Extract Color Palette from Base64 Image",
    description="""Analyze a base64 encoded image to extract its dominant color palette.
    
    Features:
    - Identify up to 10 dominant colors
    - Return RGB, HEX, and human-readable color names
    - Determine primary and background colors
    
    Supported image formats: PNG, JPEG, BMP
    """,
    responses={
        200: {"description": "Successful color extraction"},
        400: {"description": "Invalid base64 image"},
        500: {"description": "Internal server error during color analysis"}
    })
async def extract_colors_base64(
    request: Base64ImageRequest,
    n_colors: int = Query(5, ge=1, le=10, description="Number of dominant colors to extract (1-10). Default is 5.")
) -> ColorExtractionResponse:
    """Extract dominant colors from a base64 encoded image
    
    Analyzes the base64 encoded image and returns a detailed color palette
    with RGB, HEX, and named colors.
    
    Args:
        request: Base64 image request object
        n_colors: Number of colors to extract (1-10)
    
    Returns:
        Comprehensive color palette details
    """
    # Decode base64 image
    try:
        # Add padding if needed
        base64_image = request.base64_image.strip()
        padding_needed = len(base64_image) % 4
        if padding_needed:
            base64_image += '=' * (4 - padding_needed)
        
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
    
    # Open image with Pillow
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not process the image")
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Extract color palette
    try:
        color_analysis = ColorExtractor.analyze_palette(image_array, n_colors)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error analyzing color palette")
    
    return color_analysis

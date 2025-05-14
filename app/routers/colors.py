import base64
import io

from typing import List, Dict, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query
from pydantic import BaseModel
from PIL import Image
import numpy as np

from app.services.color_extractor import ColorExtractor
from app.utils.image_validator import validate_image

router = APIRouter(prefix="/colors", tags=["Colors"])

class ColorExtractionResponse(BaseModel):
    """Response model for color extraction
    
    Provides a comprehensive breakdown of colors extracted from an image.
    Each color includes RGB, HEX, and a human-readable name.
    
    Attributes:
        colors (List[Dict]): List of extracted colors with details
        primary_color (Dict): The most dominant color in the image
        background_color (Dict): The color representing the background
    
    Example:
        {
            "colors": [
                {"rgb": [255, 0, 0], "hex": "#FF0000", "name": "red"},
                {"rgb": [0, 0, 255], "hex": "#0000FF", "name": "blue"},
                ...
            ],
            "primary_color": {"rgb": [255, 0, 0], "hex": "#FF0000", "name": "red"},
            "background_color": {"rgb": [0, 0, 255], "hex": "#0000FF", "name": "blue"}
        }
    """
    colors: List[Dict[str, Union[str, List[int]]]]
    primary_color: Dict[str, Union[str, List[int]]]
    background_color: Dict[str, Union[str, List[int]]]

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Open image with Pillow
    try:
        image = Image.open(io.BytesIO(validated_file))
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

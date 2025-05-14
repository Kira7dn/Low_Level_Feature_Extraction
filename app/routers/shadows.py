from fastapi import APIRouter, UploadFile, File, HTTPException, status
from ..services.image_processor import ImageProcessor
from ..services.shadow_analyzer import ShadowAnalyzer
from ..utils.image_validator import validate_image

router = APIRouter()

@router.post(
    "/extract-shadows",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Extract shadow level",
    response_description="Shadow intensity level (High/Moderate/Low)"
)
async def extract_shadows(file: UploadFile = File(...)):
    """
    Extract shadow intensity level from an image.
    
    - **file**: Image file (PNG, JPEG, BMP)
    
    Returns a JSON object with:
    - **shadow_level**: Shadow intensity level ('High', 'Moderate', 'Low')
    
    Example response:
    ```json
    {
      "shadow_level": "Moderate"
    }
    ```
    """
    try:
        # Validate and load image
        image_bytes = await validate_image(file)
        cv_image = ImageProcessor.load_cv2_image(image_bytes)
        
        # Analyze shadow level
        level = ShadowAnalyzer.analyze_shadow_level(cv_image)
        return {"shadow_level": level}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error analyzing shadow level: {str(e)}"
        )

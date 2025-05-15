from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.image_processor import ImageProcessor
from ..services.shadow_analyzer import ShadowAnalyzer
from ..utils.error_handler import validate_image

router = APIRouter(prefix="/shadows", tags=["Shadows"])

@router.post(
    "/extract",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Extract shadow level",
    response_description="Shadow intensity level (High/Moderate/Low)"
)
async def detect_shadows(file: UploadFile = File(...)):
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
        try:
            cv_image = ImageProcessor.load_cv2_image(image_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error processing image: {str(e)}"
            )
        # Analyze shadow level
        try:
            # Use ShadowAnalyzer to detect shadow intensity
            shadow_analysis = ShadowAnalyzer.analyze_shadow_level(cv_image)
            print(f"Shadow analysis result: {shadow_analysis}")
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error analyzing shadows: {str(e)}"
            )
        
        return {"shadow_level": shadow_analysis}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error analyzing shadow level: {str(e)}"
        )

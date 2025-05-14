from fastapi import APIRouter, UploadFile, File, HTTPException, status
from ..services.image_processor import ImageProcessor
from ..utils.error_handler import validate_image

router = APIRouter(prefix='/shapes', tags=['Shapes'])

@router.post('/detect')
async def detect_shapes(
    file: UploadFile = File(..., description='Image file to analyze for shapes')
):
    """
    Detect and analyze shapes in an uploaded image.
    
    Args:
        file: Image file to analyze
    
    Returns:
        Dict containing detected shapes and their properties
    
    Raises:
        HTTPException: For invalid image or processing errors
    """
    try:
        # Validate the image
        validated_file = await validate_image(file)
        
        # Process the image
        cv_image = ImageProcessor.load_cv2_image(validated_file)
        
        # Detect shapes (placeholder for actual shape detection)
        try:
            shape_analysis = {
                'detected_shapes': ['rectangle', 'circle'],
                'shape_count': 2
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error analyzing shapes: {str(e)}"
            )
        
        return shape_analysis
    
    except HTTPException as e:
        # Re-raise HTTPExceptions from image validation
        raise e
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error processing image: {str(e)}"
        )

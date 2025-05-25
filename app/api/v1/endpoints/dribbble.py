from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional

from app.services.dribble.dribbble_extractor import DribbbleExtractor
from app.api.v1.models.dribble import DesignInfo

router = APIRouter(
    prefix="/dribbble",
    tags=["dribbble"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)


@router.get("/scraper", response_model=DesignInfo)
async def get_design_info(
    url: str = Query(..., description="URL of the Dribbble design page")
) -> DesignInfo:
    """
    Get design information from a Dribbble URL.

    Args:
        url: URL of the Dribbble design (e.g., https://dribbble.com/shots/12107985-Hospital-Dashboard-UI-Kit)

    Returns:
        DesignInfo: Detailed information about the design

    Raises:
        HTTPException: If the URL is invalid or extraction fails
    """
    try:
        return await DribbbleExtractor.extract_design_info(url)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}",
        )

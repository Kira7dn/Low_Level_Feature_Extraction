from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """
    Endpoint to expose Prometheus metrics.
    
    Returns:
        Response: Prometheus metrics in plain text format
    """
    return Response(
        content=generate_latest(), 
        media_type=CONTENT_TYPE_LATEST
    )

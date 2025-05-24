"""
Main router configuration for the analyze endpoints.

This module sets up the main FastAPI router and includes all the route modules.
"""

from fastapi import APIRouter, status

# Import route modules
from . import routes  # noqa: F401

# Create the main router
router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)

# Include the routes from the routes module
router.include_router(routes.router)

"""
Analyze router package.

This package contains all the routes and logic for the image analysis API.
"""

# Import the router to make it available when importing from the package
from .router import router

__all__ = ["router"]

import logging
import traceback
from typing import Dict, Any
from fastapi import HTTPException
import imghdr
import io

class ValidationException(Exception):
    """Custom exception for validation errors"""
    def __init__(self, detail: str, error_code: str = "validation_error"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)

class StorageException(Exception):
    """Custom exception for storage-related errors"""
    def __init__(self, detail: str, error_code: str = "storage_error"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)

class AuthenticationException(Exception):
    """Custom exception for authentication errors"""
    def __init__(self, detail: str, error_code: str = "auth_error"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)

class ResourceNotFoundException(Exception):
    """Custom exception for resource not found errors"""
    def __init__(self, detail: str, error_code: str = "not_found"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)

class BusinessLogicException(Exception):
    """Custom exception for business logic violations"""
    def __init__(self, detail: str, error_code: str = "business_logic_error"):
        self.detail = detail
        self.error_code = error_code
        super().__init__(self.detail)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_error(exception: Exception, context: Dict[str, Any] = None):
    """
    Log errors with optional context
    
    Args:
        exception (Exception): The exception to log
        context (Dict[str, Any], optional): Additional context for the error
    """
    error_details = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc()
    }
    
    if context:
        error_details["context"] = context
    
    logger.error(f"Error occurred: {error_details}")

def validate_image(file):
    """Simple image validation for personal use"""
    # Read file contents
    file_contents = file.file.read()
    file.file.seek(0)  # Reset file pointer
    
    # Check file type
    file_type = imghdr.what(io.BytesIO(file_contents))
    if file_type not in ['jpeg', 'png', 'jpg']:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    # Check file size (optional)
    file_size = len(file_contents)
    if file_size > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="File too large")
    
    return file_contents

def upload_error_handler(exception: Exception):
    """
    Handle upload-related errors
    
    Args:
        exception (Exception): The exception that occurred during upload
    
    Returns:
        Dict[str, Any]: Standardized error response
    """
    log_error(exception)
    
    if isinstance(exception, ValidationException):
        return {
            "error": {
                "code": exception.error_code,
                "message": exception.detail
            }
        }
    elif isinstance(exception, StorageException):
        return {
            "error": {
                "code": exception.error_code,
                "message": exception.detail
            }
        }
    else:
        return {
            "error": {
                "code": "unknown_error",
                "message": "An unexpected error occurred during upload"
            }
        }

def global_exception_handler(request, exc):
    """
    Global exception handler for FastAPI
    
    Args:
        request: The request object
        exc: The exception that occurred
    
    Returns:
        JSONResponse with error details
    """
    from fastapi.responses import JSONResponse
    
    log_error(exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "server_error",
                "message": "An internal server error occurred"
            }
        }
    )

def validation_exception_handler(request, exc):
    """
    Handler for validation exceptions
    
    Args:
        request: The request object
        exc: The validation exception
    
    Returns:
        JSONResponse with validation error details
    """
    from fastapi.responses import JSONResponse
    
    log_error(exc)
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": str(exc)
            }
        }
    )

def http_exception_handler(request, exc):
    """
    Handler for HTTP exceptions
    
    Args:
        request: The request object
        exc: The HTTP exception
    
    Returns:
        JSONResponse with HTTP error details
    """
    from fastapi.responses import JSONResponse
    
    log_error(exc)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": exc.detail
            }
        }
    )

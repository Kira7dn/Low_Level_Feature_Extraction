import logging
import traceback
from typing import Dict, Any
from fastapi import HTTPException, UploadFile
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

async def validate_image(file):
    # Ensure file is provided
    if not file:
        raise HTTPException(
            status_code=400, 
            detail={"error": {"message": "No file uploaded", "code": "NO_FILE"}}
        )
    
    # Ensure file has expected attributes
    if not hasattr(file, 'filename') or not hasattr(file, 'file'):
        raise HTTPException(
            status_code=400, 
            detail={"error": {"message": "Invalid file type", "code": "INVALID_FILE_TYPE"}}
        )
    
    # Debug logging
    logger.info(f"Validating file: {file.filename}, Content-Type: {file.content_type}")
    
    # Predefined validation configurations
    VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    VALID_CONTENT_TYPES = ['image/png', 'image/jpeg', 'image/jpg']
    VALID_IMGHDR_TYPES = ['jpeg', 'png', 'jpg']
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    # Validate filename and content type
    filename = file.filename.lower() if file.filename else ''
    content_type = file.content_type.lower() if file.content_type else ''
    """
    Validate uploaded image file with comprehensive checks.
    
    Args:
        file (UploadFile): Uploaded file to validate
    
    Raises:
        HTTPException: For various validation failures
    
    Returns:
        bytes: Validated file contents
    """
    # Predefined validation configurations
    VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    VALID_CONTENT_TYPES = ['image/png', 'image/jpeg', 'image/jpg']
    VALID_IMGHDR_TYPES = ['jpeg', 'png', 'jpg']
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    # Check if file is provided
    if not file:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": "No file uploaded. Please provide an image file.",
                    "code": "NO_FILE_UPLOADED"
                }
            }
        )
    
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=400, 
            detail="Invalid filename. Filename cannot be empty.",
            headers={"X-Error-Code": "INVALID_FILENAME"}
        )
    
    # Normalize filename and content type
    filename = file.filename.lower()
    content_type = file.content_type.lower() if file.content_type else ''
    
    # Validate file extension
    if not any(filename.endswith(ext) for ext in VALID_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": f"Unsupported image format. Supported formats: {', '.join(ext[1:] for ext in VALID_EXTENSIONS)}",
                    "code": "INVALID_IMAGE_TYPE"
                }
            }
        )
    
    # Validate content type
    if content_type and content_type not in VALID_CONTENT_TYPES:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": f"Unsupported file type. Supported types: {', '.join(VALID_CONTENT_TYPES)}",
                    "code": "INVALID_CONTENT_TYPE"
                }
            }
        )
    
    # Read file contents
    try:
        file_contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": f"Error reading file: {str(e)}",
                    "code": "FILE_READ_ERROR"
                }
            }
        )
    
    # Verify image type using imghdr
    try:
        file_type = imghdr.what(io.BytesIO(file_contents))
        if not file_type or file_type not in VALID_IMGHDR_TYPES:
            raise ValueError("Unsupported image type")
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": "Unsupported image format. Only PNG and JPEG are supported.",
                    "code": "INVALID_IMAGE_TYPE"
                }
            }
        )
    
    # Check file size
    file_size = len(file_contents)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail={
                "error": {
                    "message": f"File too large. Maximum file size is {MAX_FILE_SIZE/1024/1024:.1f}MB.",
                    "code": "FILE_TOO_LARGE"
                }
            }
        )
    
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

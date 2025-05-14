from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routers import colors, text, shapes, base, shadows, fonts


app = FastAPI(
    title="Low-Level Feature Extraction API",
    description="API for extracting design elements from images",
    version="1.0.0"
)

# Register exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)

# Optional: Register specific custom exception handlers
app.add_exception_handler(ValidationException, global_exception_handler)
app.add_exception_handler(AuthenticationException, global_exception_handler)
app.add_exception_handler(ResourceNotFoundException, global_exception_handler)
app.add_exception_handler(StorageException, global_exception_handler)
app.add_exception_handler(BusinessLogicException, global_exception_handler)

# Include routers
app.include_router(base.router)
app.include_router(colors.router, tags=["colors"])
app.include_router(base.router, tags=["base"])
app.include_router(shadows.router, tags=["shadows"])
app.include_router(fonts.router, tags=["fonts"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Low-Level Feature Extraction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

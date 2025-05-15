from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from app.routers import colors, text, shapes, shadows, fonts, metrics, unified_analysis
from app.utils.error_handler import validate_image

def create_app():
    global app
    app = FastAPI(
        title="Low-Level Feature Extraction API",
        description="API for extracting design elements from images",
        version="1.0.0"
    )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)}
        )

    # Register routers
    app.include_router(colors.router, tags=["colors"])
    app.include_router(text.router, tags=["text"])
    app.include_router(shapes.router, tags=["shapes"])
    app.include_router(shadows.router, tags=["shadows"])
    app.include_router(fonts.router, tags=["fonts"])
    app.include_router(metrics.router, tags=["metrics"])
    app.include_router(unified_analysis.router, tags=["unified_analysis"])

    @app.get("/")
    async def root():
        return {"message": "Welcome to the Low-Level Feature Extraction API"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app

# Global app instance for backwards compatibility
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

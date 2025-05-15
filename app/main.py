import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routers import colors, text, shapes, shadows, fonts, metrics, unified_analysis
from app.utils.error_handler import validate_image
from app.routers.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

def create_app():
    global app
    app = FastAPI(
        title="Low-Level Feature Extraction API",
        description="API for extracting design elements from images",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(method=method, endpoint=path, status=response.status_code).inc()
            REQUEST_LATENCY.labels(endpoint=path).observe(time.time() - start_time)
            return response
        except Exception as e:
            ERROR_COUNT.labels(endpoint=path, error_type=type(e).__name__).inc()
            raise

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        ERROR_COUNT.labels(
            endpoint=request.url.path,
            error_type=type(exc).__name__
        ).inc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": type(exc).__name__
            }
        )

    # Register routers
    app.include_router(colors.router, prefix="/api/v1", tags=["colors"])
    app.include_router(text.router, prefix="/api/v1", tags=["text"])
    app.include_router(shapes.router, prefix="/api/v1", tags=["shapes"])
    app.include_router(shadows.router, prefix="/api/v1", tags=["shadows"])
    app.include_router(fonts.router, prefix="/api/v1", tags=["fonts"])
    app.include_router(metrics.router, tags=["monitoring"])
    app.include_router(unified_analysis.router, prefix="/api/v1", tags=["unified_analysis"])

    @app.get("/")
    async def root():
        return {
            "message": "Welcome to the Low-Level Feature Extraction API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "metrics_url": "/metrics"
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app

# Global app instance for backwards compatibility
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

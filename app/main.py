from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routers import colors, text, shapes, shadows, fonts, unified_analysis
from app.core.config import settings

def create_app():
    global app
    
    # Configure logging
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger = logging.getLogger(__name__)
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="API for extracting design elements from images",
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
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
    app.include_router(unified_analysis.router, prefix="/api/v1", tags=["unified_analysis"])

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_content = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Feature Extraction API</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        line-height: 1.6;
                    }
                    h1 { color: #2c3e50; }
                    .links a {
                        display: inline-block;
                        margin: 10px;
                        padding: 10px 20px;
                        background: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }
                    .links a:hover { background: #2980b9; }
                </style>
            </head>
            <body>
                <h1>Feature Extraction API</h1>
                <p>Welcome to the Low-Level Feature Extraction API. This service provides image analysis capabilities including:</p>
                <ul>
                    <li>Color Analysis</li>
                    <li>Text Recognition</li>
                    <li>Shape Detection</li>
                    <li>Shadow Analysis</li>
                    <li>Font Detection</li>
                </ul>
                <div class="links">
                    <a href="/docs">API Documentation</a>
                    <a href="/api/v1/unified-analysis">Try Unified Analysis</a>
                </div>
                <p><small>Version: 1.0.0</small></p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    return app

# Global app instance for backwards compatibility
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

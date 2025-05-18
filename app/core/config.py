from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Low-Level Feature Extraction API"
    
    # Image Processing Settings
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    IMAGE_MAX_WIDTH = 1920
    IMAGE_MAX_HEIGHT = 1080
    IMAGE_QUALITY = 85
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Low-Level Feature Extraction API"
    
    # Image Processing Settings
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()

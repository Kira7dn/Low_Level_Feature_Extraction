from pydantic_settings import BaseSettings
from typing import List, ClassVar, Optional
from typing_extensions import Annotated

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Low-Level Feature Extraction API"
    
    # Image Processing Settings
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    IMAGE_DOWNLOAD_TIMEOUT: int = 30  # seconds
    IMAGE_MAX_WIDTH: int = 1920
    IMAGE_MAX_HEIGHT: int = 1080
    IMAGE_QUALITY: int = 85
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # API Keys (for compatibility with crawl4ai)
    ANTHROPIC_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    
    class Config:
        case_sensitive = True
        extra = "allow"  # Allow extra fields to be present in environment variables
        env_file = ".env"

# Create settings instance
settings = Settings()

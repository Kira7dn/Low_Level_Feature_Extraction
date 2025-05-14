import os
from typing import Literal, Optional
from pydantic import BaseSettings, validator, Field, RedisDsn

class CacheSettings(BaseSettings):
    """Configuration for caching mechanism"""
    # Cache type selection
    CACHE_TYPE: Literal['memory', 'redis'] = 'memory'
    
    # In-memory cache settings
    MEMORY_CACHE_MAX_SIZE: int = 100
    MEMORY_CACHE_TTL: int = 300  # 5 minutes
    
    # Redis cache settings
    REDIS_URL: Optional[RedisDsn] = None
    REDIS_CACHE_TTL: int = 600  # 10 minutes
    
    # Specific endpoint caching configurations
    COLOR_EXTRACTION_CACHE_TTL: int = 600  # 10 minutes
    TEXT_EXTRACTION_CACHE_TTL: int = 300  # 5 minutes
    
    @validator('REDIS_URL', pre=True)
    def set_redis_url(cls, v):
        """Validate and set Redis URL"""
        return v or os.getenv('REDIS_URL', 'redis://localhost:6379/0')

class Settings(BaseSettings):
    """Main application settings"""
    # API settings
    API_TITLE: str = "Low-Level Feature Extraction API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "0") == "1"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Image processing settings
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    MAX_IMAGE_DIMENSION: int = 4000
    OPTIMIZATION_DIMENSION: int = 1200
    
    # Caching settings (embedded cache configuration)
    CACHE: CacheSettings = CacheSettings()
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        # Allow population from environment variables
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Allow arbitrary types for complex configurations
        arbitrary_types_allowed = True

# Create global settings instance
settings = Settings()

# Expose cache settings for easy access
cache_settings = settings.CACHE

def get_cache_config():
    """
    Dynamically generate cache configuration based on current settings
    
    :return: Dictionary of cache configuration parameters
    """
    return {
        'cache_type': cache_settings.CACHE_TYPE,
        'max_size': cache_settings.MEMORY_CACHE_MAX_SIZE,
        'ttl': cache_settings.MEMORY_CACHE_TTL,
        'redis_url': cache_settings.REDIS_URL
    }

# Endpoint-specific cache TTL mappings
ENDPOINT_CACHE_CONFIGS = {
    'color_extraction': cache_settings.COLOR_EXTRACTION_CACHE_TTL,
    'text_extraction': cache_settings.TEXT_EXTRACTION_CACHE_TTL
}

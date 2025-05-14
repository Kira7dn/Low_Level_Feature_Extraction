import os
import json
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    pass

class ConfigurationManager:
    """
    Centralized configuration management system
    
    Features:
    - Environment-specific configuration
    - Secure secrets management
    - Configuration validation
    - Caching for performance
    - Flexible configuration sources
    """
    
    # Configuration sources priority
    _CONFIG_SOURCES = [
        'environment_variables',
        'config_files',
        'default_values'
    ]
    
    # Supported environments
    ENVIRONMENTS = {
        'development': 'dev',
        'testing': 'test',
        'staging': 'stage',
        'production': 'prod'
    }
    
    def __init__(self, 
                 config_dir: Optional[str] = None, 
                 env: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir (str, optional): Directory containing configuration files
            env (str, optional): Deployment environment
        """
        # Determine configuration directory
        self._config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), 
            '..'
        )
        
        # Determine environment
        self._env = env or os.getenv('APP_ENV', 'development').lower()
        
        # Validate environment
        if self._env not in self.ENVIRONMENTS:
            raise ConfigurationError(f"Invalid environment: {self._env}")
        
        # Setup logging
        self._logger = logging.getLogger(__name__)
    
    @lru_cache(maxsize=32)
    def get_config(self, config_key: str) -> Any:
        """
        Retrieve configuration value with multi-source lookup
        
        Args:
            config_key (str): Configuration key to retrieve
        
        Returns:
            Any: Configuration value
        
        Raises:
            ConfigurationError: If configuration key is not found
        """
        # Check environment variables first
        env_var = f"APP_{config_key.upper()}"
        env_value = os.getenv(env_var)
        if env_value is not None:
            return self._parse_value(env_value)
        
        # Check configuration files
        config_file_path = os.path.join(
            self._config_dir, 
            f'config.{self.ENVIRONMENTS[self._env]}.json'
        )
        
        try:
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
                if config_key in config_data:
                    return config_data[config_key]
        except FileNotFoundError:
            self._logger.warning(f"Configuration file not found: {config_file_path}")
        except json.JSONDecodeError:
            self._logger.error(f"Invalid JSON in configuration file: {config_file_path}")
        
        # Check default configurations
        default_configs = self._get_default_configs()
        if config_key in default_configs:
            return default_configs[config_key]
        
        raise ConfigurationError(f"Configuration key not found: {config_key}")
    
    def _parse_value(self, value: str) -> Any:
        """
        Parse configuration value with type inference
        
        Args:
            value (str): Raw configuration value
        
        Returns:
            Any: Parsed configuration value
        """
        # Convert string to appropriate type
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def _get_default_configs(self) -> Dict[str, Any]:
        """
        Retrieve default configuration values
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        return {
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'MAX_WORKERS': 4,
            'CACHE_TTL': 300,
            'IMAGE_MAX_SIZE': 10 * 1024 * 1024,  # 10 MB
            'PERFORMANCE_LOG_RETENTION': 1000,
            'ALLOWED_IMAGE_FORMATS': ['png', 'jpg', 'webp']
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration integrity
        
        Returns:
            bool: Whether configuration is valid
        """
        try:
            # Perform validation checks
            debug_mode = self.get_config('DEBUG')
            log_level = self.get_config('LOG_LEVEL')
            max_workers = self.get_config('MAX_WORKERS')
            
            # Additional validation logic
            assert isinstance(debug_mode, bool)
            assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            assert 1 <= max_workers <= 16
            
            return True
        except Exception as e:
            self._logger.error(f"Configuration validation failed: {e}")
            return False

# Create global configuration manager instance
config = ConfigurationManager()

# Convenience functions for common configurations
def get_config(key: str) -> Any:
    """Retrieve configuration value"""
    return config.get_config(key)

def is_debug_mode() -> bool:
    """Check if application is in debug mode"""
    return get_config('DEBUG')

def get_log_level() -> str:
    """Get configured log level"""
    return get_config('LOG_LEVEL')

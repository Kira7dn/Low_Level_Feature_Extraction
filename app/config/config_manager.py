import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging
import jsonschema

# Load environment variables
load_dotenv()

class ConfigurationManager:
    """
    Centralized configuration management system with environment-specific 
    configuration and secure secrets handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        :param config_path: Optional path to configuration file
        """
        # Default configuration paths
        self._default_config_paths = [
            os.path.join(os.path.dirname(__file__), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        ]
        
        # Configuration schema for validation
        self._config_schema = {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "environment": {"type": "string"},
                "debug": {"type": "boolean"},
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                        "file": {"type": "string"}
                    }
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "number"},
                        "name": {"type": "string"}
                    }
                },
                "secrets": {
                    "type": "object",
                    "properties": {
                        "database_password": {"type": "string"},
                        "api_key": {"type": "string"}
                    }
                }
            },
            "required": ["app_name", "environment"]
        }
        
        # Load configuration
        self._config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment
        
        :param config_path: Optional path to configuration file
        :return: Loaded configuration dictionary
        """
        # Prioritize passed config path, then default paths
        config_paths = [config_path] + self._default_config_paths if config_path else self._default_config_paths
        
        for path in config_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as config_file:
                        config = json.load(config_file)
                        
                        # Override with environment variables
                        config = self._override_with_env_vars(config)
                        
                        # Validate configuration
                        self._validate_config(config)
                        
                        return config
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Could not load config from {path}: {e}")
        
        # Fallback to environment variables if no config file found
        return self._get_env_config()
    
    def _override_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration with environment variables
        
        :param config: Original configuration dictionary
        :return: Updated configuration dictionary
        """
        # Environment variable overrides
        env_overrides = {
            'APP_NAME': ['app_name'],
            'APP_ENV': ['environment'],
            'DEBUG': ['debug'],
            'DB_HOST': ['database', 'host'],
            'DB_PORT': ['database', 'port'],
            'DB_NAME': ['database', 'name'],
            'DB_PASSWORD': ['secrets', 'database_password'],
            'API_KEY': ['secrets', 'api_key']
        }
        
        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate through nested dictionary
                current = config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                current[config_path[-1]] = env_value
        
        return config
    
    def _get_env_config(self) -> Dict[str, Any]:
        """
        Generate configuration from environment variables
        
        :return: Configuration dictionary from environment
        """
        return {
            "app_name": os.getenv("APP_NAME", "Low_Level_Feature_Extraction"),
            "environment": os.getenv("APP_ENV", "development"),
            "debug": os.getenv("DEBUG", "False").lower() == "true",
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", 5432)),
                "name": os.getenv("DB_NAME", "feature_extraction")
            },
            "secrets": {
                "database_password": os.getenv("DB_PASSWORD"),
                "api_key": os.getenv("API_KEY")
            }
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against predefined schema
        
        :param config: Configuration dictionary to validate
        :raises jsonschema.ValidationError: If configuration is invalid
        """
        try:
            jsonschema.validate(instance=config, schema=self._config_schema)
        except jsonschema.ValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """
        Setup logging based on configuration
        """
        log_level = getattr(logging, self._config.get('logging', {}).get('level', 'INFO').upper())
        log_file = self._config.get('logging', {}).get('file')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        :param key: Dot-separated configuration key
        :param default: Default value if key not found
        :return: Configuration value
        """
        current = self._config
        for k in key.split('.'):
            if isinstance(current, dict):
                current = current.get(k, default)
            else:
                return default
        return current
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Securely retrieve secrets
        
        :param key: Secret key
        :return: Secret value
        """
        secrets = self._config.get('secrets', {})
        secret = secrets.get(key)
        
        # Optional: Add additional security measures like encryption
        return secret
    
    def is_production(self) -> bool:
        """
        Check if current environment is production
        
        :return: True if production, False otherwise
        """
        return self._config.get('environment', '').lower() == 'production'

# Create a singleton instance
config_manager = ConfigurationManager()

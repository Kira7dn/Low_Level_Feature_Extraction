from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CDNConfig:
    """
    Configuration management for CDN providers
    Supports multiple CDN providers with flexible configuration
    """
    
    # Supported CDN providers
    PROVIDERS = {
        'cloudflare': {
            'base_url': 'https://cdn.cloudflare.com',
            'transformation_support': ['resize', 'format', 'quality']
        },
        'cloudinary': {
            'base_url': 'https://res.cloudinary.com',
            'transformation_support': ['resize', 'format', 'quality', 'crop']
        },
        'imgix': {
            'base_url': 'https://assets.imgix.net',
            'transformation_support': ['resize', 'format', 'quality', 'fit']
        }
    }
    
    @classmethod
    def get_cdn_config(cls, provider: str = 'cloudflare') -> Dict[str, Any]:
        """
        Retrieve CDN configuration for a specific provider
        
        Args:
            provider (str, optional): Name of the CDN provider. Defaults to 'cloudflare'.
        
        Returns:
            Dict[str, Any]: CDN configuration details
        
        Raises:
            ValueError: If an unsupported CDN provider is specified
        """
        if provider.lower() not in cls.PROVIDERS:
            raise ValueError(f"Unsupported CDN provider: {provider}. Supported providers: {list(cls.PROVIDERS.keys())}")
        
        # Retrieve provider-specific configuration
        config = cls.PROVIDERS[provider.lower()].copy()
        
        # Add environment-specific overrides
        config['api_key'] = os.getenv(f'{provider.upper()}_CDN_API_KEY', '')
        config['account_id'] = os.getenv(f'{provider.upper()}_CDN_ACCOUNT_ID', '')
        
        return config

    @classmethod
    def validate_cdn_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate CDN configuration
        
        Args:
            config (Dict[str, Any]): CDN configuration to validate
        
        Returns:
            bool: Whether the configuration is valid
        """
        required_keys = ['base_url', 'transformation_support']
        return all(key in config for key in required_keys)

# Example usage in environment configuration
def get_default_cdn_provider():
    """
    Retrieve the default CDN provider from environment variables
    
    Returns:
        str: Default CDN provider name
    """
    provider = os.getenv('DEFAULT_CDN_PROVIDER', 'cloudflare').lower()
    
    # Validate provider exists
    if provider not in CDNConfig.PROVIDERS:
        # Fallback to cloudflare if invalid provider specified
        provider = 'cloudflare'
    
    return provider

# Get default CDN provider and configuration
DEFAULT_CDN_PROVIDER = get_default_cdn_provider()
DEFAULT_CDN_CONFIG = CDNConfig.get_cdn_config(DEFAULT_CDN_PROVIDER)

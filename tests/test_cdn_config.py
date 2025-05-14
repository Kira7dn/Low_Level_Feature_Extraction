import os
import pytest
from app.config.cdn_config import CDNConfig

def test_get_cdn_config_default():
    """Test default CDN configuration retrieval"""
    config = CDNConfig.get_cdn_config()
    
    assert config['base_url'] == 'https://cdn.cloudflare.com'
    assert 'resize' in config['transformation_support']
    assert 'format' in config['transformation_support']
    assert 'quality' in config['transformation_support']

def test_get_cdn_config_specific_provider():
    """Test CDN configuration for specific providers"""
    providers = ['cloudflare', 'cloudinary', 'imgix']
    
    for provider in providers:
        config = CDNConfig.get_cdn_config(provider)
        
        assert 'base_url' in config
        assert 'transformation_support' in config
        assert len(config['transformation_support']) > 0

def test_get_cdn_config_invalid_provider():
    """Test error handling for unsupported CDN providers"""
    with pytest.raises(ValueError, match="Unsupported CDN provider"):
        CDNConfig.get_cdn_config('unsupported_provider')

def test_validate_cdn_config():
    """Test CDN configuration validation"""
    valid_config = {
        'base_url': 'https://example.com',
        'transformation_support': ['resize', 'format']
    }
    
    invalid_config_1 = {'base_url': 'https://example.com'}
    invalid_config_2 = {'transformation_support': ['resize']}
    
    assert CDNConfig.validate_cdn_config(valid_config) is True
    assert CDNConfig.validate_cdn_config(invalid_config_1) is False
    assert CDNConfig.validate_cdn_config(invalid_config_2) is False

def test_environment_variable_override(monkeypatch):
    """Test environment variable overrides for CDN configuration"""
    # Test valid provider override
    monkeypatch.setenv('DEFAULT_CDN_PROVIDER', 'cloudinary')
    
    from importlib import reload
    import app.config.cdn_config
    reload(app.config.cdn_config)
    
    from app.config.cdn_config import DEFAULT_CDN_PROVIDER, DEFAULT_CDN_CONFIG
    
    assert DEFAULT_CDN_PROVIDER == 'cloudinary'
    assert DEFAULT_CDN_CONFIG['base_url'] == 'https://res.cloudinary.com'
    
    # Test invalid provider override (should fallback to cloudflare)
    monkeypatch.setenv('DEFAULT_CDN_PROVIDER', 'invalid_provider')
    
    reload(app.config.cdn_config)
    from app.config.cdn_config import DEFAULT_CDN_PROVIDER, DEFAULT_CDN_CONFIG
    
    assert DEFAULT_CDN_PROVIDER == 'cloudflare'
    assert DEFAULT_CDN_CONFIG['base_url'] == 'https://cdn.cloudflare.com'

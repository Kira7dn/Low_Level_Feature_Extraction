import os
import json
import tempfile
import pytest
from app.config.settings import ConfigurationManager, ConfigurationError

class TestConfigurationManager:
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_default_configuration(self):
        """Test retrieval of default configuration values"""
        config_manager = ConfigurationManager()
        
        # Test some default configurations
        assert config_manager.get_config('DEBUG') is False
        assert config_manager.get_config('LOG_LEVEL') == 'INFO'
        assert config_manager.get_config('MAX_WORKERS') == 4
    
    def test_environment_variable_override(self, monkeypatch):
        """Test configuration override via environment variables"""
        # Set environment variable
        monkeypatch.setenv('APP_DEBUG', 'true')
        monkeypatch.setenv('APP_LOG_LEVEL', 'DEBUG')
        
        config_manager = ConfigurationManager()
        
        assert config_manager.get_config('DEBUG') is True
        assert config_manager.get_config('LOG_LEVEL') == 'DEBUG'
    
    def test_config_file_override(self, temp_config_dir):
        """Test configuration override via config files"""
        # Create a test configuration file
        test_config_path = os.path.join(temp_config_dir, 'config.dev.json')
        with open(test_config_path, 'w') as f:
            json.dump({
                'DEBUG': True,
                'LOG_LEVEL': 'DEBUG',
                'MAX_WORKERS': 8
            }, f)
        
        # Create configuration manager with temp config dir
        config_manager = ConfigurationManager(
            config_dir=temp_config_dir, 
            env='development'
        )
        
        assert config_manager.get_config('DEBUG') is True
        assert config_manager.get_config('LOG_LEVEL') == 'DEBUG'
        assert config_manager.get_config('MAX_WORKERS') == 8
    
    def test_config_value_parsing(self):
        """Test parsing of different configuration value types"""
        config_manager = ConfigurationManager()
        
        # Test boolean parsing
        assert config_manager._parse_value('true') is True
        assert config_manager._parse_value('false') is False
        
        # Test integer parsing
        assert config_manager._parse_value('42') == 42
        
        # Test float parsing
        assert config_manager._parse_value('3.14') == 3.14
        
        # Test string parsing
        assert config_manager._parse_value('test_string') == 'test_string'
    
    def test_invalid_environment(self):
        """Test handling of invalid environment"""
        with pytest.raises(ConfigurationError, match="Invalid environment"):
            ConfigurationManager(env='invalid_env')
    
    def test_missing_config_key(self):
        """Test retrieval of missing configuration key"""
        config_manager = ConfigurationManager()
        
        with pytest.raises(ConfigurationError, match="Configuration key not found"):
            config_manager.get_config('NON_EXISTENT_KEY')
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config_manager = ConfigurationManager()
        
        # Validate configuration
        assert config_manager.validate_config() is True
    
    def test_lru_cache_behavior(self):
        """Test LRU caching of configuration values"""
        config_manager = ConfigurationManager()
        
        # First call should compute and cache
        first_call = config_manager.get_config('DEBUG')
        
        # Subsequent calls should return cached value
        second_call = config_manager.get_config('DEBUG')
        
        assert first_call == second_call

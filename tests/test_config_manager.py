import os
import unittest
import tempfile
import json
from app.config.config_manager import ConfigurationManager

class TestConfigurationManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        self.temp_config_file = tempfile.mktemp()
        test_config = {
            "app_name": "TestApp",
            "environment": "test",
            "debug": True,
            "database": {
                "host": "testhost",
                "port": 5432,
                "name": "testdb"
            },
            "secrets": {
                "database_password": "test_password",
                "api_key": "test_api_key"
            }
        }
        
        with open(self.temp_config_file, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        # Remove temporary config file
        if os.path.exists(self.temp_config_file):
            os.unlink(self.temp_config_file)
    
    def test_config_loading(self):
        # Test loading from specific config file
        config_manager = ConfigurationManager(self.temp_config_file)
        
        # Check basic configuration values
        self.assertEqual(config_manager.get('app_name'), 'TestApp')
        self.assertEqual(config_manager.get('environment'), 'test')
        self.assertTrue(config_manager.get('debug'))
    
    def test_nested_config_retrieval(self):
        config_manager = ConfigurationManager(self.temp_config_file)
        
        # Test nested configuration retrieval
        self.assertEqual(config_manager.get('database.host'), 'testhost')
        self.assertEqual(config_manager.get('database.port'), 5432)
    
    def test_secret_retrieval(self):
        config_manager = ConfigurationManager(self.temp_config_file)
        
        # Test secret retrieval
        self.assertEqual(config_manager.get_secret('database_password'), 'test_password')
        self.assertEqual(config_manager.get_secret('api_key'), 'test_api_key')
    
    def test_environment_override(self):
        # Set environment variables to override config
        os.environ['APP_NAME'] = 'OverrideApp'
        os.environ['APP_ENV'] = 'production'
        os.environ['DEBUG'] = 'false'
        
        try:
            config_manager = ConfigurationManager(self.temp_config_file)
            
            # Check if environment variables override config
            self.assertEqual(config_manager.get('app_name'), 'OverrideApp')
            self.assertEqual(config_manager.get('environment'), 'production')
            self.assertFalse(config_manager.get('debug'))
        finally:
            # Clean up environment variables
            del os.environ['APP_NAME']
            del os.environ['APP_ENV']
            del os.environ['DEBUG']
    
    def test_is_production(self):
        # Test production environment detection
        config_manager = ConfigurationManager(self.temp_config_file)
        self.assertFalse(config_manager.is_production())
        
        # Simulate production environment
        os.environ['APP_ENV'] = 'production'
        try:
            config_manager = ConfigurationManager(self.temp_config_file)
            self.assertTrue(config_manager.is_production())
        finally:
            del os.environ['APP_ENV']

if __name__ == '__main__':
    unittest.main()

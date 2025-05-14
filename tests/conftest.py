import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
import asyncio

# Ensure the app module can be imported
import app

# Modify Python import system to help with imports
sys.path.append(os.path.join(project_root, 'app'))

# Add tests directory to Python path
sys.path.append(os.path.dirname(__file__))

# Configure async event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

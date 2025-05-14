import asyncio
import time
import pytest
import pytest_asyncio

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the asyncio event loop"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
from fastapi.testclient import TestClient
from httpx import AsyncClient
import sys
import os

# Determine the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Ensure the project root is in PYTHONPATH
if project_root not in sys.path:
    sys.path.append(project_root)
import base64
import io
from PIL import Image
import numpy as np

from app.main import create_app

# Create an app instance for testing
app = create_app()
from app.utils.cache import SimpleCache, cache_result

# Utility function to create a test image
def create_test_image():
    """Create a simple test image"""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

@pytest.mark.asyncio
async def test_simple_cache():
    """Test the SimpleCache implementation"""
    cache = SimpleCache(max_size=2, ttl=1)
    
    # Test basic set and get
    cache.set('key1', 'value1')
    assert cache.get('key1') == 'value1'
    
    # Test cache expiration
    await asyncio.sleep(2)
    assert cache.get('key1') is None
    
    # Test max size
    cache.set('key2', 'value2')
    cache.set('key3', 'value3')
    cache.set('key4', 'value4')
    
    # Oldest item should be removed
    assert cache.get('key2') is None
    assert cache.get('key3') is not None
    assert cache.get('key4') is not None

@pytest.mark.asyncio
async def test_cache_result_decorator():
    """Test the cache_result decorator functionality"""
    # Mock function to test caching
    call_count = 0
    
    @cache_result(ttl=2)
    async def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call should compute and cache
    result1 = await expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call should return cached result
    result2 = await expensive_function(5)
    assert result2 == 10
    assert call_count == 1
    
    # Wait for cache to expire
    await asyncio.sleep(3)
    
    # Call after expiration should recompute
    result3 = await expensive_function(5)
    assert result3 == 10
    assert call_count == 2

@pytest.mark.asyncio
async def test_color_extraction_caching():
    """Test caching of color extraction endpoints"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create a test image
        test_image = create_test_image()
        
        # First call to extract colors
        start_time1 = time.time()
        response1 = await client.post("/colors/extract-base64", json={
            "base64_image": test_image,
            "n_colors": 5
        })
        end_time1 = time.time()
        assert response1.status_code == 200
        first_result = response1.json()
        first_call_time = end_time1 - start_time1
        
        # Second call with same image (should be cached)
        start_time2 = time.time()
        response2 = await client.post("/colors/extract-base64", json={
            "base64_image": test_image,
            "n_colors": 5
        })
        end_time2 = time.time()
        assert response2.status_code == 200
        second_result = response2.json()
        second_call_time = end_time2 - start_time2
        
        # Verify results are identical
        assert first_result == second_result
        
        # Cached call should be significantly faster
        # Allow some variance, but cached call should be at least 5x faster
        assert second_call_time < (first_call_time / 5)

def test_cache_invalidation():
    """Test cache invalidation mechanisms"""
    cache = SimpleCache(max_size=10, ttl=60)
    
    # Set initial value
    cache.set('test_key', 'initial_value')
    assert cache.get('test_key') == 'initial_value'
    
    # Clear entire cache
    cache.clear()
    assert cache.get('test_key') is None

# Performance and load testing
@pytest.mark.performance
@pytest.mark.asyncio
async def test_cache_performance_under_load():
    """Simulate concurrent requests to test caching performance"""
    async def make_color_extraction_request(image):
        async with AsyncClient(app=app, base_url="http://test") as client:
            return await client.post("/colors/extract-base64", json={
                "base64_image": image,
                "n_colors": 5
            })
    
    # Create multiple test images
    test_images = [create_test_image() for _ in range(5)]
    
    # Simulate concurrent requests
    start_time = time.time()
    tasks = [make_color_extraction_request(img) for img in test_images * 10]
    responses = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Verify all requests were successful
    assert all(response.status_code == 200 for response in responses)
    
    # Total execution time should be significantly less than sequential processing
    print(f"Concurrent requests total time: {end_time - start_time} seconds")

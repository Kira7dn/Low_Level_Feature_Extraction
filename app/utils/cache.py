from functools import wraps
import hashlib
import json
import time
import logging
from typing import Dict, Any, Callable, Optional
import redis
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCache:
    def __init__(self, max_size=100, ttl=300):
        """Initialize cache with max size and time-to-live (seconds)"""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Any:
        """Get item from cache if it exists and is not expired"""
        if key not in self.cache:
            return None
        
        item = self.cache[key]
        if time.time() > item['expires']:
            # Remove expired item
            del self.cache[key]
            return None
        
        return item['value']
    
    def set(self, key: str, value: Any) -> None:
        """Add item to cache with expiration"""
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['expires'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + self.ttl
        }
    
    def clear(self) -> None:
        """Clear all cache items"""
        self.cache.clear()

class CacheManager:
    """
    Advanced caching manager with support for in-memory and Redis caching
    """
    def __init__(self, 
                 cache_type: str = 'memory', 
                 max_size: int = 100, 
                 ttl: int = 300, 
                 redis_url: Optional[str] = None):
        """
        Initialize cache manager with configurable caching strategy
        
        :param cache_type: Type of cache ('memory', 'redis')
        :param max_size: Maximum number of items in cache
        :param ttl: Default time-to-live for cache entries
        :param redis_url: Redis connection URL
        """
        self.cache_type = cache_type
        self.max_size = max_size
        self.ttl = ttl
        
        # In-memory cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Redis cache
        self._redis_client = None
        if cache_type == 'redis':
            try:
                self._redis_client = redis.Redis.from_url(
                    redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                )
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}")
                # Fallback to memory cache
                self.cache_type = 'memory'
    
    def get(self, key: str) -> Any:
        """
        Retrieve item from cache
        
        :param key: Cache key
        :return: Cached value or None
        """
        try:
            if self.cache_type == 'redis' and self._redis_client:
                value = self._redis_client.get(key)
                return json.loads(value) if value else None
            
            # In-memory cache
            if key not in self._memory_cache:
                return None
            
            item = self._memory_cache[key]
            if time.time() > item["expires"]:
                del self._memory_cache[key]
                return None
            
            return item["value"]
        
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Add item to cache
        
        :param key: Cache key
        :param value: Value to cache
        :param ttl: Optional time-to-live override
        """
        try:
            expires = time.time() + (ttl or self.ttl)
            
            if self.cache_type == 'redis' and self._redis_client:
                # Redis caching
                self._redis_client.setex(
                    key, 
                    ttl or self.ttl, 
                    json.dumps(value)
                )
                return
            
            # In-memory cache management
            if len(self._memory_cache) >= self.max_size:
                # Remove oldest item
                oldest_key = min(
                    self._memory_cache.keys(), 
                    key=lambda k: self._memory_cache[k]["expires"]
                )
                del self._memory_cache[oldest_key]
            
            self._memory_cache[key] = {
                "value": value,
                "expires": expires
            }
        
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    def clear(self) -> None:
        """Clear all cache items"""
        try:
            if self.cache_type == 'redis' and self._redis_client:
                self._redis_client.flushdb()
            else:
                self._memory_cache.clear()
            
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

def cache_result(ttl: int = 300, 
                 cache_type: str = 'memory', 
                 max_size: int = 100):
    """
    Decorator to cache function results with configurable caching
    
    :param ttl: Time-to-live for cache entries
    :param cache_type: Type of cache ('memory', 'redis')
    :param max_size: Maximum cache size
    """
    cache_manager = CacheManager(
        cache_type=cache_type, 
        max_size=max_size, 
        ttl=ttl
    )
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add args and kwargs to key
            for arg in args:
                if hasattr(arg, "read"):
                    # For file-like objects, hash the content
                    pos = arg.tell()
                    content = await arg.read()
                    await arg.seek(pos)
                    key_parts.append(hashlib.md5(content).hexdigest())
                else:
                    key_parts.append(str(arg))
            
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = hashlib.md5(json.dumps(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function if not in cache
            result = await func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result)
            logger.info(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

# Global cache instances
memory_cache = CacheManager(cache_type='memory')
redis_cache = CacheManager(cache_type='redis')

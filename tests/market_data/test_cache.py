"""Tests for cache implementations."""
import pytest
import asyncio
import os
import shutil
import time
from typing import Optional

from src.market_data.cache.disk import DiskCache
from src.market_data.models.common import CacheConfig
from src.market_data.exceptions import CacheError

@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary directory for cache files."""
    cache_path = tmp_path / "test_cache"
    cache_path.mkdir()
    yield str(cache_path)
    # Cleanup
    shutil.rmtree(str(cache_path))

@pytest.fixture
def cache_config(cache_dir):
    return CacheConfig(
        enabled=True,
        ttl=1,  # 1 second TTL for testing
        directory=cache_dir
    )

@pytest.fixture
def disk_cache(cache_config):
    return DiskCache(cache_config)

@pytest.mark.asyncio
async def test_basic_cache_operations(disk_cache):
    """Test basic set/get/delete operations."""
    # Set value
    await disk_cache.set("test_key", {"data": "test_value"})
    
    # Get value
    result = await disk_cache.get("test_key")
    assert result == {"data": "test_value"}
    
    # Delete value
    await disk_cache.delete("test_key")
    result = await disk_cache.get("test_key")
    assert result is None

@pytest.mark.asyncio
async def test_cache_expiration(disk_cache):
    """Test cache entry expiration."""
    # Set value with 1 second TTL
    await disk_cache.set("test_key", "test_value", ttl=1)
    
    # Should be available immediately
    result = await disk_cache.get("test_key")
    assert result == "test_value"
    
    # Wait for expiration
    await asyncio.sleep(1.1)
    
    # Should return None after expiration
    result = await disk_cache.get("test_key")
    assert result is None

@pytest.mark.asyncio
async def test_cache_clear(disk_cache):
    """Test clearing all cache entries."""
    # Set multiple values
    test_data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    for key, value in test_data.items():
        await disk_cache.set(key, value)
        
    # Verify values are cached
    for key, value in test_data.items():
        result = await disk_cache.get(key)
        assert result == value
        
    # Clear cache
    await disk_cache.clear()
    
    # Verify all values are cleared
    for key in test_data:
        result = await disk_cache.get(key)
        assert result is None

@pytest.mark.asyncio
async def test_cache_large_values(disk_cache):
    """Test caching of large values that trigger compression."""
    # Create large string that will exceed compression threshold
    large_value = "x" * (20 * 1024)  # 20KB
    
    # Cache large value
    await disk_cache.set("large_key", large_value)
    
    # Retrieve and verify
    result = await disk_cache.get("large_key")
    assert result == large_value

@pytest.mark.asyncio
async def test_cache_invalid_keys(disk_cache):
    """Test cache behavior with invalid keys."""
    invalid_keys = [
        "key/with/slashes",
        "key with spaces",
        "key@with#special$chars",
        "../path/traversal/attempt",
        "very" * 100 + "long"  # Very long key
    ]
    
    # Should handle all keys without error
    for key in invalid_keys:
        await disk_cache.set(key, "test_value")
        result = await disk_cache.get(key)
        assert result == "test_value"
        await disk_cache.delete(key)

@pytest.mark.asyncio
async def test_concurrent_cache_access(disk_cache):
    """Test concurrent cache access."""
    async def cache_operation(i: int):
        key = f"key_{i}"
        value = f"value_{i}"
        
        # Set value
        await disk_cache.set(key, value)
        
        # Small delay to increase chance of concurrent access
        await asyncio.sleep(0.01)
        
        # Get value
        result = await disk_cache.get(key)
        assert result == value
        
        # Delete value
        await disk_cache.delete(key)
        
        return True
    
    # Run multiple operations concurrently
    tasks = [cache_operation(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert all(results)

@pytest.mark.asyncio
async def test_cache_complex_objects(disk_cache):
    """Test caching of complex nested objects."""
    test_data = {
        "string": "test",
        "number": 123,
        "float": 123.456,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "nested": {
            "dict": {"key": "value"},
            "list": [{"nested": "item"}]
        }
    }
    
    # Cache complex object
    await disk_cache.set("complex", test_data)
    
    # Retrieve and verify
    result = await disk_cache.get("complex")
    assert result == test_data

@pytest.mark.asyncio
async def test_cache_persistence(cache_config):
    """Test cache persistence across instances."""
    # Create first cache instance
    cache1 = DiskCache(cache_config)
    await cache1.set("persistent", "test_value")
    
    # Create second cache instance
    cache2 = DiskCache(cache_config)
    result = await cache2.get("persistent")
    assert result == "test_value"

def test_cache_directory_creation(tmp_path):
    """Test cache directory is created if it doesn't exist."""
    cache_dir = str(tmp_path / "nonexistent")
    assert not os.path.exists(cache_dir)
    
    config = CacheConfig(directory=cache_dir)
    cache = DiskCache(config)
    
    assert os.path.exists(cache_dir)
    assert os.path.isdir(cache_dir)
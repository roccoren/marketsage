"""Disk-based cache implementation."""
import os
import json
import asyncio
import aiofiles
import time
from typing import Optional, Any
from dataclasses import asdict
import gzip

from ..interfaces import Cache
from ..models.common import CacheConfig
from ..exceptions import CacheError

class DiskCache(Cache):
    """
    Disk-based cache implementation with compression support.
    
    Features:
    - Configurable TTL for cached items
    - Automatic compression for large responses
    - Thread-safe file operations
    - Efficient storage and retrieval
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize disk cache with configuration.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_dir = config.directory or "./cache"
        self.compress_threshold = 1024 * 10  # Compress if > 10KB
        self._lock = asyncio.Lock()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Generate filesystem path for cache key."""
        # Convert key to filename-safe format
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self.cache_dir, f"{safe_key}.cache")

    async def _read_cache_file(self, path: str) -> Optional[tuple[Any, float]]:
        """Read and parse cache file."""
        try:
            if not os.path.exists(path):
                return None
                
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
                
            # Check if content is compressed
            if content.startswith(b'\x1f\x8b'):
                content = gzip.decompress(content)
                
            data = json.loads(content.decode('utf-8'))
            return data['value'], data['expires_at']
            
        except Exception as e:
            raise CacheError(f"Failed to read cache file: {str(e)}")

    async def _write_cache_file(
        self,
        path: str,
        value: Any,
        expires_at: float
    ) -> None:
        """Write value to cache file."""
        try:
            data = {
                'value': value,
                'expires_at': expires_at
            }
            
            # Convert to JSON and encode
            content = json.dumps(data).encode('utf-8')
            
            # Compress if content is large
            if len(content) > self.compress_threshold:
                content = gzip.compress(content)
                
            async with aiofiles.open(path, 'wb') as f:
                await f.write(content)
                
        except Exception as e:
            raise CacheError(f"Failed to write cache file: {str(e)}")

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        async with self._lock:
            path = self._get_cache_path(key)
            
            try:
                result = await self._read_cache_file(path)
                if result is None:
                    return None
                    
                value, expires_at = result
                
                # Check if expired
                if expires_at and time.time() > expires_at:
                    await self.delete(key)
                    return None
                    
                return value
                
            except Exception as e:
                # Log error but don't raise - treat as cache miss
                print(f"Cache read error: {str(e)}")
                return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        async with self._lock:
            path = self._get_cache_path(key)
            
            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
                
            await self._write_cache_file(path, value, expires_at)

    async def delete(self, key: str) -> None:
        """
        Remove a value from cache.
        
        Args:
            key: Cache key to remove
        """
        async with self._lock:
            path = self._get_cache_path(key)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                raise CacheError(f"Failed to delete cache file: {str(e)}")

    async def clear(self) -> None:
        """Remove all cached values."""
        async with self._lock:
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except Exception as e:
                raise CacheError(f"Failed to clear cache: {str(e)}")

    def __del__(self):
        """Cleanup on deletion."""
        # Nothing to clean up for disk cache
        pass
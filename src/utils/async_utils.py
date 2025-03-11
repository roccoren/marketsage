"""Async utilities for handling concurrent operations and rate limiting."""
import asyncio
import aiofiles
import aiohttp
import time
import logging
import json
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from aiohttp import TCPConnector
from functools import partial
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

# Global thread pool for operations with optimal size
thread_pool = ThreadPoolExecutor(max_workers=max(10, cpu_count() * 2))

class TokenBucket:
    """Token bucket rate limiter for controlling request rates."""
    
    def __init__(self, rate_limit_ms: int, max_tokens: int = 1):
        self.rate_limit = rate_limit_ms / 1000.0
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            new_tokens = int(time_passed / self.rate_limit)
            
            if new_tokens > 0:
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)
                self.last_update = now
            
            if self.tokens <= 0:
                wait_time = self.rate_limit * 0.1
                await asyncio.sleep(wait_time)
            else:
                self.tokens -= 1

class CircuitBreaker:
    """Circuit breaker for protecting against API failures."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"
        self._lock = asyncio.Lock()
        
    async def __call__(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
                    
            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                raise e
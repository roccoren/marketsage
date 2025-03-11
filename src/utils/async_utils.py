"""Async utilities for handling concurrent operations and rate limiting."""
# Import yfinance config module first to ensure settings are applied
from . import yfinance_config

import asyncio
import aiofiles
import aiohttp
import time
import logging
import json
import os
from typing import Dict, Any, Optional, List
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pandas as pd
from aiohttp import TCPConnector
from functools import partial
import gzip
import io
import urllib3
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

# Additional yfinance configuration
yf.set_tz_cache_location("./data/yfinance_cache")

# Global thread pool for yfinance operations with optimal size
thread_pool = ThreadPoolExecutor(max_workers=max(30, cpu_count() * 4))

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

class UpdateProgressTracker:
    """Track and persist progress of stock updates."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.progress_file = f"{data_dir}/update_progress.json"
        self.progress: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
    async def load_progress(self) -> None:
        """Load saved progress data if it exists."""
        try:
            async with aiofiles.open(self.progress_file, 'r') as f:
                self.progress = json.loads(await f.read())
        except:
            self.progress = {
                'last_update': None,
                'exchanges': {},
                'failed_symbols': []
            }
                
    async def update_exchange_progress(
        self,
        exchange: str,
        total_symbols: int,
        completed_symbols: int
    ) -> None:
        """Update progress for an exchange."""
        async with self._lock:
            self.progress['exchanges'][exchange] = {
                'total': total_symbols,
                'completed': completed_symbols,
                'last_update': time.time()
            }
            await self._save_progress()
                
    async def add_failed_symbol(self, exchange: str, symbol: str) -> None:
        """Record a failed symbol update."""
        async with self._lock:
            self.progress['failed_symbols'].append({
                'exchange': exchange,
                'symbol': symbol,
                'timestamp': time.time()
            })
            await self._save_progress()
                
    async def _save_progress(self) -> None:
        """Save current progress to file."""
        try:
            async with aiofiles.open(self.progress_file, 'w') as f:
                await f.write(json.dumps(self.progress, indent=2))
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")

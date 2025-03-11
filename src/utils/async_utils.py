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
import re
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# Additional yfinance configuration
try:
    # Use absolute path for cache location
    cache_dir = os.path.abspath("./data/yfinance_cache")
    os.makedirs(cache_dir, exist_ok=True)
    yf.set_tz_cache_location(cache_dir)
except Exception as e:
    logger.warning(f"Failed to set TzCache location: {e}")

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

class YahooFinanceClient:
    """Optimized async client for Yahoo Finance API."""
    
    def __init__(self, rate_limit_ms: int = 2000, batch_size: int = None, data_dir: str = None):
        # Dynamic batch size based on system resources
        if batch_size is None:
            optimal_batch = cpu_count() * 2
            self.batch_size = min(50, max(16, optimal_batch))
        else:
            self.batch_size = min(batch_size, 50)
        
        self.rate_limiter = TokenBucket(max(rate_limit_ms, 500), 20)
        self.session = None
        self.data_dir = data_dir
        if self.data_dir:
            logger.info(f"Data will be stored in: {self.data_dir}")
        self._lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker()
        self.chunk_size = 1000
        
        # NYSE symbols that are known to be problematic with YFinance
        self.known_problematic_symbols = {
            'A', 'AA', 'AAM', 'AAM.W', 'AACT.W'  # Add more as you discover them
        }
        
        # Cache for successful symbol formats
        self.symbol_format_cache = {}
        
    async def __aenter__(self):
        if not self.session:
            connector = TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=120
            )
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    def _optimize_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> tuple[str, str]:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime('%Y-%m-%d')
            
        return start_date, end_date

    def _clean_symbol(self, symbol: str) -> str:
        """Clean and normalize stock symbol for Yahoo Finance API.
        
        Args:
            symbol: Raw stock symbol from exchange listing
            
        Returns:
            Cleaned symbol compatible with Yahoo Finance
        """
        if not symbol or not isinstance(symbol, str):
            return ""
            
        # Check cache first for previously successful formats
        if symbol in self.symbol_format_cache:
            return self.symbol_format_cache[symbol]
            
        # Remove leading/trailing whitespace
        symbol = symbol.strip()
        
        # Remove leading $ if present
        if symbol.startswith('$'):
            symbol = symbol[1:]
            
        # Handle known problematic symbols
        if symbol in self.known_problematic_symbols:
            if symbol == 'A':
                return 'A.US'  # Try alternative format
            if symbol == 'AA':
                return 'AA.US'  # Try alternative format
            if symbol.endswith('.W'):  # Warrants
                base = symbol.split('.')[0]
                return f"{base}-WT"
        
        # Handle NYSE special cases
        if '.' in symbol:
            # Handle warrants, preferred stocks, and other special securities
            parts = symbol.split('.')
            base = parts[0]
            ext = parts[1]
            
            if ext == 'W':  # Warrants need special handling
                # For warrants, Yahoo uses -WT suffix
                return f"{base}-WT"
                
            if ext == 'P' or ext.startswith('P'):  # Preferred stocks
                # For preferred stocks, Yahoo uses -P suffix
                return f"{base}-{ext}"
                
            if ext in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
                # For share classes, Yahoo uses dot notation
                return f"{base}.{ext}"
                
            # Handle exchange-specific extensions
            if ext in ['HK', 'AX']:
                if ext == 'HK':
                    # Hong Kong stocks format: Format as 4 digits + .HK
                    base_symbol = ''.join(c for c in base if c.isdigit())
                    if len(base_symbol) < 4:
                        base_symbol = base_symbol.zfill(4)
                    return f"{base_symbol}.{ext}"
                return f"{base}.{ext}"  # Keep original format for other exchanges
        
        # Remove unsupported characters (Yahoo Finance doesn't handle some special chars)
        clean_symbol = re.sub(r'[^A-Za-z0-9\.\-]', '', symbol)
        
        return clean_symbol
            
    def _transform_to_history_format(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            history = []
            for i in range(0, len(data), self.chunk_size):
                chunk = data.iloc[i:i + self.chunk_size]
                chunk_entries = self._process_data_chunk(chunk)
                history.extend(chunk_entries)
            
            return {
                "history": history
            }
        except Exception as e:
            logger.error(f"Error transforming data to history format: {e}")
            return None
    
    def _process_data_chunk(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        entries = []
        for date, row in chunk.iterrows():
            entry = {
                "date": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            }
            entries.append(entry)
        return entries

    async def _load_raw_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return json.loads(content)
            else:
                logger.debug(f"No existing raw JSON file found at {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading raw JSON data: {e}")
            return None

    def _save_raw_json(self, data: Dict[str, Any], file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved raw data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            
    async def _fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical stock data with improved error handling and retry mechanism."""
        if not symbol:
            logger.error("Empty symbol provided")
            return None
        
        logger.debug(f"Fetching data for symbol: {symbol}")
        
        # Skip known invalid/delisted symbols
        if symbol.strip() in self.known_problematic_symbols:
            logger.warning(f"Skipping known problematic symbol: {symbol}")
            return None
        
        # Try up to 3 different symbol formats if needed
        symbol_formats = [
            self._clean_symbol(symbol),  # Standard cleaning
            symbol.strip(),              # Raw symbol
            f"{symbol.strip()}.US"       # .US suffix format
        ]
        
        # Remove duplicates and empty strings
        symbol_formats = [fmt for fmt in list(dict.fromkeys(symbol_formats)) if fmt]
        
        if not symbol_formats:
            logger.error(f"No valid symbol format found for {symbol}")
            return None
            
        last_error = None
        
        # Try each symbol format with retries
        for format_idx, symbol_fmt in enumerate(symbol_formats):
            for attempt in range(3):  # 3 retries per format
                try:
                    def fetch():
                        try:
                            # Use requests session with custom headers for better success
                            session = requests.Session()
                            session.headers.update({
                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                'Accept-Language': 'en-US,en;q=0.5',
                                'Accept-Encoding': 'gzip, deflate, br',
                                'DNT': '1',
                                'Connection': 'keep-alive',
                                'Upgrade-Insecure-Requests': '1'
                            })
                            
                            # Create ticker with custom session
                            ticker = yf.Ticker(symbol_fmt, session=session)
                            data = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                            
                            if not data.empty:
                                logger.info(f"Successfully fetched data for {symbol} (as {symbol_fmt}): {len(data)} rows")
                                # Cache successful format for future use
                                self.symbol_format_cache[symbol] = symbol_fmt
                                return data
                            else:
                                logger.warning(f"Empty data returned for {symbol} (as {symbol_fmt})")
                                return None
                                
                        except Exception as e:
                            logger.error(f"Error fetching {symbol} (as {symbol_fmt}): {str(e)}")
                            raise
                    
                    # Run in thread pool with timeout
                    future = asyncio.get_event_loop().run_in_executor(thread_pool, fetch)
                    try:
                        # Impose timeout to avoid hanging
                        result = await asyncio.wait_for(future, timeout=15)
                        if result is not None and not result.empty:
                            return result
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout fetching {symbol} (as {symbol_fmt})")
                        # Try to cancel the future
                        future.cancel()
                        raise TimeoutError(f"Fetch timeout for {symbol_fmt}")
                
                except (RequestException, TimeoutError, ValueError) as e:
                    last_error = e
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"Attempt {attempt+1} failed for {symbol} (as {symbol_fmt}): {str(e)}. Waiting {wait_time}s.")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    last_error = e
                    logger.error(f"Unexpected error fetching {symbol} (as {symbol_fmt}): {str(e)}")
                    break  # Don't retry on unknown errors
        
        if last_error:
            logger.error(f"All attempts failed for {symbol}: {str(last_error)}")
        
        return None

    async def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self.rate_limiter.acquire()
        
        try:
            def fetch_info():
                clean_symbol = self._clean_symbol(symbol)
                ticker = yf.Ticker(clean_symbol)
                try:
                    market_cap = ticker.info.get('marketCap')
                except:
                    market_cap = None
                
                return {
                    'price': {
                        'quoteType': 'equity',
                        'marketState': 'regular',
                        'tradeable': True,
                        'marketCap': market_cap
                    }
                }
            
            return await asyncio.get_event_loop().run_in_executor(thread_pool, fetch_info)
        except Exception as e:
            logger.error(f"Error fetching ticker info for {symbol}: {str(e)}")
            return None

    async def get_history_batch(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        incremental: bool = False
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        async def protected_batch_fetch():
            end_date_opt = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
            results = {}
            
            semaphore = asyncio.Semaphore(8)
            tasks = []
            
            async def process_symbol(symbol: str):
                async with semaphore:
                    await self.rate_limiter.acquire()
                    if not symbol or not isinstance(symbol, str):
                        return symbol, None
                    
                    symbol = symbol.strip()
                    if not symbol:
                        return symbol, None
                    
                    symbol_start_date, symbol_end_date = self._optimize_date_range(start_date, end_date_opt)
                    data = await self._fetch_data(symbol, symbol_start_date, symbol_end_date)
                    
                    if data is None or data.empty:
                        return symbol, None
                    
                    result = self._transform_to_history_format(data)
                    if result and self.data_dir:
                        try:
                            json_path = os.path.join(self.data_dir, f"{symbol}.json")
                            self._save_raw_json(result, json_path)
                        except Exception as e:
                            logger.error(f"Error saving data for {symbol}: {e}")
                    
                    return symbol, result
            
            for symbol in symbols:
                tasks.append(process_symbol(symbol))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, tuple):
                    results[result[0]] = result[1]
            
            return results
            
        return await self.circuit_breaker(protected_batch_fetch)

    async def get_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        incremental: bool = False
    ) -> Optional[Dict[str, Any]]:
        results = await self.get_history_batch([symbol], start_date, end_date, incremental)
        return results.get(symbol)

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
"""Async utilities for handling concurrent operations and rate limiting."""
import asyncio
import aiofiles
import aiohttp
import time
import logging
import json
from typing import Dict, Any, Optional, List
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pandas as pd
from aiohttp import TCPConnector
from functools import partial
import gzip
import io

logger = logging.getLogger(__name__)

# Global thread pool for yfinance operations
thread_pool = ThreadPoolExecutor(max_workers=20)  # Increased worker count

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
    
    def __init__(self, rate_limit_ms: int, batch_size: int):
        self.rate_limiter = TokenBucket(max(rate_limit_ms, 1000), 10)  # Increased token capacity
        self.batch_size = min(batch_size, 100)  # Cap batch size at 100
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker()
        
    async def __aenter__(self):
        if not self.session:
            # Optimize connection pooling and enable keepalive
            connector = TCPConnector(
                limit=100,  # Maximum number of concurrent connections
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                use_dns_cache=True,
                keepalive_timeout=60  # Keep connections alive for 60 seconds
            )
            
            # Optimize headers for better network performance
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    def _optimize_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> tuple[str, str]:
        """Optimize date range to minimize data transfer."""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            # Default to 1 year of data if no start date specified
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime('%Y-%m-%d')
            
        return start_date, end_date

    async def _fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Optimized data fetching using thread pool."""
        try:
            logger.debug(f"Fetching data for symbol: {symbol}")
            def fetch():
                ticker = yf.Ticker(symbol)
                logger.debug(f"Created Ticker object for {symbol}")
                return ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
            
            return await asyncio.get_event_loop().run_in_executor(thread_pool, fetch)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def _is_valid_stock_data(self, info: Optional[Dict[str, Any]]) -> bool:
        """Validate if the stock data is for an active equity/ETF."""
        if not info or 'price' not in info:
            return False
            
        price_data = info['price']
        # Most fields from Yahoo Finance might be missing, so we'll be more lenient
        quote_type = price_data.get('quoteType', '').lower()
        if quote_type and quote_type not in {'equity', 'etf', 'stock', ''}:
            return False
            
        # Only check market state if it's provided
        market_state = price_data.get('marketState', '').lower()
        if market_state and market_state not in {'regular', 'pre', 'post', 'closed', 'postpost', 'prepre'}:
            return False
            
        # Don't check tradeable status since it might not be provided
        return True
        
    async def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a ticker."""
        await self.rate_limiter.acquire()
        
        try:
            def fetch_info():
                ticker = yf.Ticker(symbol)
                return {
                    'price': {
                        'quoteType': 'equity',  # Default to equity since YF might not provide this
                        'marketState': 'regular',  # Default to regular since we're using daily data
                        'tradeable': True  # Default to true since these are major stocks
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
        end_date: Optional[str] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get historical data for multiple symbols with optimized batch processing."""
        async def protected_batch_fetch():
            start_date_opt, end_date_opt = self._optimize_date_range(start_date, end_date)
            results = {}
            
            # Use semaphore to control concurrent operations
            semaphore = asyncio.Semaphore(20)  # Increased from 10 to 20
            
            async def process_symbol(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
                async with semaphore:
                    await self.rate_limiter.acquire()
                    
                    data = await self._fetch_data(symbol, start_date_opt, end_date_opt)
                    if data is None or data.empty:
                        return symbol, None
                    
                    # Optimize data structure creation
                    timestamps = [int(ts.timestamp()) for ts in data.index]
                    quotes = data[['Open', 'High', 'Low', 'Close', 'Volume']].values.T
                    
                    result = {
                        'chart': {
                            'result': [{
                                'timestamp': timestamps,
                                'indicators': {
                                    'quote': [{
                                        'open': quotes[0].tolist(),
                                        'high': quotes[1].tolist(),
                                        'low': quotes[2].tolist(),
                                        'close': quotes[3].tolist(),
                                        'volume': quotes[4].tolist()
                                    }],
                                    'adjclose': [{
                                        'adjclose': quotes[3].tolist()  # Using adjusted close
                                    }]
                                }
                            }]
                        }
                    }
                    return symbol, result
            
            # Process symbols in optimized batches
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                tasks = [process_symbol(symbol) for symbol in batch]
                
                # Gather results with error handling
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out any exceptions and add successful results
                for result in batch_results:
                    if isinstance(result, tuple):
                        results[result[0]] = result[1]
            
            return results

        # Use circuit breaker for batch fetching
        return await self.circuit_breaker(protected_batch_fetch)

    async def get_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get historical data for a single symbol."""
        results = await self.get_history_batch([symbol], start_date, end_date)
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

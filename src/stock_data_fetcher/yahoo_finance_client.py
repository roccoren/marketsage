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
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
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
        if not symbol or not isinstance(symbol, str):
            return ""
            
        if symbol.startswith('$'):
            symbol = symbol[1:]
            
        if symbol.endswith('.HK'):
            base_symbol = symbol.split('.')[0]
            base_symbol = ''.join(c for c in base_symbol if c.isdigit())
            if len(base_symbol) < 4:
                base_symbol = base_symbol.zfill(4)
            return f"{base_symbol}.HK"
            
        if '.AX' in symbol:
            base_symbol = symbol.split('.')[0]
            return f"{base_symbol}.AX"
        
        return symbol
            
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
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None
                
            logger.debug(f"Fetching data for symbol: {symbol}")
            
            def fetch():
                clean_symbol = self._clean_symbol(symbol)
                ticker = yf.Ticker(clean_symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                if not data.empty:
                    logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
                    return data
                return None
            
            return await asyncio.get_event_loop().run_in_executor(thread_pool, fetch)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
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

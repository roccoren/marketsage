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

class YahooFinanceClient:
    """Optimized async client for Yahoo Finance API."""
    
    def __init__(self, rate_limit_ms: int = 2000, batch_size: int = None, data_dir: str = None, exchange: str = None):
        # Dynamic batch size based on system resources
        if batch_size is None:
            # Use CPU count to determine optimal batch size
            optimal_batch = cpu_count() * 2
            self.batch_size = min(50, max(16, optimal_batch))  # Increased caps
        else:
            self.batch_size = min(batch_size, 50)  # Increased upper limit

        self.rate_limiter = TokenBucket(max(rate_limit_ms, 500), 20)  # Increased concurrency
        self.session: Optional[aiohttp.ClientSession] = None
        self.data_dir = data_dir
        self.exchange = exchange
        if self.data_dir:
            logger.info(f"Data will be stored in: {self.data_dir}")
        self._lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker()
        self.chunk_size = 1000  # Number of rows to process at once
        
    async def __aenter__(self):
        if not self.session:
            # Optimize connection pooling and enable keepalive with increased limits
            connector = TCPConnector(
                limit=100,  # Increased connection pool
                limit_per_host=20,  # Increased per-host limit
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                use_dns_cache=True,
                keepalive_timeout=120  # Keep connections alive for 2 minutes
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
                timeout=aiohttp.ClientTimeout(total=60)
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

    def _clean_symbol(self, symbol: str) -> str:
        """Clean stock symbol for Yahoo Finance API."""
        # Remove any $ prefix
        if not symbol or not isinstance(symbol, str):
            return ""
            
        if symbol.startswith('$'):
            symbol = symbol[1:]
            
        # Fix HKEX symbols
        if symbol.endswith('.HK'):
            # Extract only the numeric part
            base_symbol = symbol.split('.')[0]
            # Remove any remaining non-digit characters
            base_symbol = ''.join(c for c in base_symbol if c.isdigit())
            # Pad to 4 digits if needed
            if len(base_symbol) < 4:
                base_symbol = base_symbol.zfill(4)
            return f"{base_symbol}.HK"
            
        # Fix ASX symbols that may have duplicate .AX suffix
        if '.AX' in symbol:
            # Extract base symbol without any .AX suffixes
            base_symbol = symbol.split('.')[0]
            return f"{base_symbol}.AX"
        
        return symbol
            
    def _transform_to_history_format(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Transform DataFrame to required JSON history format using chunked processing."""
        try:
            history = []
            # Process DataFrame in chunks to reduce memory usage
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
        """Process a chunk of DataFrame data."""
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
        """Load data from raw JSON file."""
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
        """Save data as raw uncompressed JSON."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as plain JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved raw data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")

    def _save_compressed_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save data as compressed JSON."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save as compressed JSON
            compressed_path = f"{file_path}.gz"
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=None)  # Remove indentation to reduce size
            
            logger.debug(f"Saved compressed data to {compressed_path}")
        except Exception as e:
            logger.error(f"Error saving compressed data: {e}")
            # Fallback to uncompressed JSON if compression fails
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

    async def _load_compressed_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load data from compressed JSON."""
        try:
            compressed_path = f"{file_path}.gz"
            if os.path.exists(compressed_path):
                async with aiofiles.open(compressed_path, 'rb') as f:
                    content = await f.read()
                    with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8') as gz:
                        return json.load(gz)
            elif os.path.exists(file_path):
                # Fallback to uncompressed file
                async with aiofiles.open(file_path, 'r') as f:
                    return json.loads(await f.read())
            return None
        except Exception as e:
            logger.error(f"Error loading compressed data: {e}")
            return None

    def _transform_to_history_format_old(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Transform DataFrame to required JSON history format."""
        try:
            history = []
            for date, row in data.iterrows():
                entry = {
                    "date": date.strftime('%Y-%m-%d'),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume'])
                }
                history.append(entry)
            
            return {
                "history": history
            }
        except Exception as e:
            logger.error(f"Error transforming data to history format: {e}")
            return None
            
    async def _fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Optimized data fetching using thread pool."""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None
            logger.debug(f"Fetching data for symbol: {symbol}")
            
            async def fetch_in_thread():
                retries = 3
                for attempt in range(retries):
                    try:
                        clean_symbol = self._clean_symbol(symbol)
                        ticker = yf.Ticker(clean_symbol)
                        if not ticker:
                            logger.error(f"Failed to create ticker for {clean_symbol}")
                            return None
                        logger.debug(f"Created Ticker object for {clean_symbol}")
                        logger.debug(f"Attempting to fetch data between {start_date} and {end_date}")

                        # Try getting history data first
                        # Special handling for HKEX stocks
                        if symbol.endswith('.HK'):
                            try:
                                # First try getting recent data
                                data = ticker.history(period='1mo', interval='1d')
                                if not data.empty:
                                    # If we can get recent data, fetch the full range
                                    logger.info(f"Successfully validated {symbol}, fetching full range")
                                    time.sleep(1)  # Small delay before full fetch
                                    full_data = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                                    if not full_data.empty:
                                        logger.info(f"Successfully fetched full data for {symbol}: {len(full_data)} rows")
                                        return full_data
                                
                                logger.debug(f"Recent data fetch failed for {symbol}, trying alternative validation")
                                info = ticker.info
                                if info.get('regularMarketPrice') or info.get('currentPrice'):
                                    logger.info(f"Validated {symbol} through market price, retrying data fetch")
                                    return ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                            except Exception as e:
                                logger.warning(f"HKEX validation failed for {symbol}: {str(e)}")
                        else:
                            # Non-HKEX stocks use standard fetch
                            data = ticker.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                            if not data.empty:
                                logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
                                return data

                    except Exception as e:
                        error_msg = str(e)
                        # Special handling for timezone errors
                        if "exchangeTimezoneName" in error_msg:
                            logger.warning(f"Timezone error for {symbol}, attempting alternative validation")
                            try:
                                # Try to get basic info without timezone dependency
                                info = ticker.info
                                logger.debug(f"Checking market price for {symbol}")
                                if info.get('regularMarketPrice'):
                                    logger.info(f"Symbol {symbol} validated through market price")
                                    # Retry with shorter time range
                                    time.sleep(1)  # Small delay before retry
                                    data = ticker.history(period='1mo', interval='1d')
                                    if not data.empty:
                                        return data
                            except Exception as inner_e:
                                logger.error(f"Alternative validation failed for {symbol}: {str(inner_e)}")
                        
                        if attempt < retries - 1:
                            wait_time = 2 * (attempt + 1)  # Progressive backoff
                            logger.info(f"Retrying {symbol} after {wait_time}s (attempt {attempt + 1})")
                            time.sleep(wait_time)
                            continue
                        
                        logger.error(f"Failed to fetch {symbol} after {retries} attempts: {error_msg}")
                        return None

                return None

            def fetch():
                return asyncio.run(fetch_in_thread())
            
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
                # Clean symbol format before creating ticker
                clean_symbol = self._clean_symbol(symbol)
                
                ticker = yf.Ticker(clean_symbol)
                # Try to get market cap directly, with fallback defaults
                try:
                    market_cap = ticker.info.get('marketCap')
                except:
                    market_cap = None
                
                return {
                    'price': {
                        'quoteType': 'equity',  # Default to equity
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
        """
        Get historical data for multiple symbols with optimized batch processing.
        When incremental=True, only fetches data from the last available date.
        """
        async def protected_batch_fetch():
            end_date_opt = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
            today = datetime.now()
            # If in incremental mode, use yesterday as the end date to ensure we get completed trading days
            if incremental and not end_date:
                yesterday = today - timedelta(days=1)
                end_date_opt = yesterday.strftime('%Y-%m-%d')
                logger.info(f"Using yesterday ({end_date_opt}) as end date for incremental update")
            
            global_end_date = end_date_opt
            results = {}
            
            # Use semaphore to control concurrent operations
            semaphore = asyncio.Semaphore(8)  # Further reduced concurrent operations

            logger.debug(f"Batch fetch date range: start={start_date}, end={end_date_opt}")
            
            async def process_symbol(symbol: str) -> tuple[str, Optional[Dict[str, Any]]]:
                async with semaphore:
                    await self.rate_limiter.acquire()

                    # Validate symbol
                    if not symbol or not isinstance(symbol, str):
                        logger.error("Empty or invalid symbol provided")
                        return symbol, None
                        
                    # Strip whitespace and validate
                    symbol = symbol.strip()
                    if not symbol:
                        return symbol, None
                    
                    # Clean symbol format
                    clean_symbol = self._clean_symbol(symbol)
                    
                    # Determine exchange directory
                    exchange_dir = None
                    if symbol.endswith('.HK'):
                        exchange_dir = 'HKEX'
                    elif symbol.endswith('.AX'):
                        exchange_dir = 'ASX'
                    else:
                        # For NYSE/NASDAQ stocks, use the exchange specified in constructor
                        if not self.exchange:
                            raise ValueError("Exchange must be specified for NYSE/NASDAQ stocks")
                        exchange_dir = self.exchange

                    # Check for existing data if incremental mode is enabled and we have a data directory
                    symbol_start_date = start_date
                    if incremental and self.data_dir:
                        # Check for an existing file
                        file_path = f"{self.data_dir}/{exchange_dir}/{symbol}.json"
                        logger.info(f"Checking for existing data at: {file_path}")
                        if os.path.exists(file_path):
                            try:
                                # Read the JSON file
                                existing_data = await self._load_raw_json(file_path)
                                
                                # Get the last timestamp
                                if existing_data.get('history', []):
                                    history = existing_data['history']
                                    if history:
                                        last_date = datetime.strptime(history[-1]['date'], '%Y-%m-%d')
                                        # Start from the next day
                                        next_day = last_date + timedelta(days=1)
                                        # Only proceed if there's new data to fetch
                                        if next_day.strftime('%Y-%m-%d') <= global_end_date:
                                            symbol_start_date = next_day.strftime('%Y-%m-%d')
                                            logger.info(f"Incremental update for {symbol}: fetching from {symbol_start_date} to {global_end_date}")
                                        
                                        result = existing_data
                                        
                                    else:
                                        logger.warning(f"No timestamps found in existing data for {symbol}")
                                else:
                                    logger.warning(f"Invalid JSON structure in existing data for {symbol}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON in existing file for {symbol}: {e}")
                            except Exception as e:
                                logger.error(f"Failed to read existing data for {symbol}: {e}")
                        else:
                            logger.info(f"No existing data found for {symbol} at {file_path}, using default start date")
                    
                    # Use default start date if we couldn't determine a specific start date
                    if not symbol_start_date:
                        symbol_start_date = start_date
                        
                    # Optimize date range (this will set default start_date if still None)    
                    symbol_start_date, symbol_end_date = self._optimize_date_range(symbol_start_date, global_end_date)
                    
                    logger.info(f"Fetching data for {clean_symbol} from {symbol_start_date} to {symbol_end_date}")
                    
                    data = await self._fetch_data(clean_symbol, symbol_start_date, symbol_end_date)
                    if data is None or data.empty:
                        logger.debug(f"No data fetched for {symbol} between {symbol_start_date} and {symbol_end_date}")
                        return symbol, None

                    def validate_history_format(data_dict):
                        """Validate the structure and content of the JSON data."""
                        try:
                            if not isinstance(data_dict, dict):
                                return False
                            history = data_dict.get('history', [])
                            if not isinstance(history, list) or not history:
                                return False
                            required_fields = {'date', 'open', 'high', 'low', 'close', 'volume'}
                            return all(isinstance(entry, dict) and
                                     all(field in entry for field in required_fields)
                                     for entry in history)
                        except Exception as e:
                            logger.error(f"History validation error for {symbol}: {e}")
                            return False

                    # Transform new data to history format
                    new_data = self._transform_to_history_format(data)
                    if new_data is None or not validate_history_format(new_data):
                        logger.error(f"Invalid new data structure for {symbol}")
                        return symbol, None

                    # If this is an incremental update and we have existing data, merge them
                    if incremental and 'result' in locals() and result is not None:
                        try:
                            # Merge the history lists while maintaining chronological order
                            combined_history = result['history'] + new_data['history']
                            json_path = f"{self.data_dir}/{exchange_dir}/{clean_symbol}.json"
                            # Remove any potential duplicates based on date
                            seen_dates = set()
                            unique_history = []
                            for entry in combined_history:
                                if entry['date'] not in seen_dates:
                                    seen_dates.add(entry['date'])
                                    unique_history.append(entry)
                            result['history'] = sorted(unique_history, key=lambda x: x['date'])
                            
                            # Save the merged data
                            try:
                                os.makedirs(f"{self.data_dir}/{exchange_dir}", exist_ok=True)
                                self._save_raw_json(result, json_path)
                                logger.info(f"Saved merged data for {symbol} to {json_path}")
                            except Exception as save_error:
                                logger.error(f"Error saving merged data for {symbol}: {save_error}")
                            return symbol, result
                        except Exception as e:
                            logger.error(f"Error merging data for {symbol}: {e}")
                    else:
                        result = new_data

                    # Additional validation check
                    if not result.get('history'):
                        logger.error(f"Missing history data for {symbol}")
                        return symbol, None
                     
                    # Log successful data processing
                    history = result['history']
                    first_date = history[0]['date'] if history else 'unknown'
                    points_count = len(history)
                    logger.info(f"Successfully processed {clean_symbol}: "
                              f"{points_count} data points from {first_date}")

                    
                    # Save data as JSON if we have a data directory
                    if self.data_dir:
                        try:
                            # Save as compressed JSON
                            # Use cleaned symbol for file name to avoid duplicate suffixes
                            json_path = f"{self.data_dir}/{exchange_dir}/{clean_symbol}.json"
                            # Ensure exchange directory exists
                            os.makedirs(f"{self.data_dir}/{exchange_dir}", exist_ok=True)
                            self._save_raw_json(result, json_path)
                            
                            # Log file size if debug is enabled
                            if logger.isEnabledFor(logging.DEBUG):
                                file_size = os.path.getsize(json_path)
                                logger.debug(f"Saved raw JSON file for {symbol}: {file_size/1024:.2f}KB")
                            logger.info(f"Saved JSON data for {clean_symbol} to {json_path}")
                        except Exception as e:
                            logger.error(f"Error saving JSON data for {symbol}: {e}")
                     
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
        end_date: Optional[str] = None,
        incremental: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get historical data for a single symbol."""
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

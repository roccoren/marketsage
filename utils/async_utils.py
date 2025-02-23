"""Async utilities for handling concurrent operations and rate limiting."""
import asyncio
import aiohttp
import aiofiles
import time
from typing import Dict, Any, Optional, List
import logging
import json

logger = logging.getLogger(__name__)

class TokenBucket:
    """Token bucket rate limiter for controlling request rates."""
    
    def __init__(self, rate_limit_ms: int, max_tokens: int = 1):
        """
        Initialize the token bucket.
        
        Args:
            rate_limit_ms: Minimum time between tokens in milliseconds
            max_tokens: Maximum number of tokens the bucket can hold
        """
        self.rate_limit = rate_limit_ms / 1000.0  # Convert to seconds
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token from the bucket, waiting if necessary."""
        async with self._lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                new_tokens = int(time_passed / self.rate_limit)
                
                if new_tokens > 0:
                    self.tokens = min(self.max_tokens, new_tokens)
                    self.last_update = now
                else:
                    await asyncio.sleep(self.rate_limit)
            
            self.tokens -= 1

class YahooFinanceClient:
    """Async client for Yahoo Finance API with rate limiting."""
    
    def __init__(self, rate_limit_ms: int, batch_size: int):
        """
        Initialize the Yahoo Finance client.
        
        Args:
            rate_limit_ms: Rate limit in milliseconds between requests
            batch_size: Maximum number of concurrent requests
        """
        self.rate_limiter = TokenBucket(max(rate_limit_ms, 10000), 3)  # Minimum 10 second delay with burst capacity of 3
        self.batch_size = batch_size
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Create client session when entering context."""
        if not self.session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Close client session when exiting context."""
        if self.session:
            await self.session.close()
            self.session = None

    def _is_valid_stock_data(self, result: Dict[str, Any]) -> bool:
        """
        Validate if the stock data indicates an active trading security.
        """
        if not result or not result.get('price'):
            logger.debug("No price data found in result")
            return False
            
        price_data = result['price']
            
        # Log validation details for debugging
        validation_checks = {
            # Check for valid market price
            'has_valid_price': isinstance(price_data.get('regularMarketPrice'), (int, float)),
            # Verify security type
            'valid_type': price_data.get('quoteType') in ['EQUITY', 'ETF'],
            # Verify the stock has recent trading activity (within last 30 days)
            'has_recent_trading': (
                isinstance(price_data.get('regularMarketTime'), (int, float)) and
                time.time() - price_data.get('regularMarketTime', 0) < 30 * 24 * 3600
            ),
            # Check market state - allow closed for after-hours
            'valid_market_state': price_data.get('marketState', '').lower() in ['pre', 'post', 'regular', 'closed'],
            # Ensure not halted
            'not_halted': price_data.get('tradingStatus', '').lower() != 'halted'
        }
            
        # Log detailed validation results
        logger.debug(f"Validation details: {validation_checks}")
        logger.debug(f"Market State: {price_data.get('marketState')}")
        logger.debug(f"Quote Type: {price_data.get('quoteType')}")
        logger.debug(f"Trading Status: {price_data.get('tradingStatus')}")
            
        return all(validation_checks.values())

    async def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information for a ticker symbol.
        
        Args:
            symbol: The stock symbol to query
            
        Returns:
            Dict containing ticker information or None if the request failed
        """
        await self.rate_limiter.acquire()
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info:
                result = {'price': info}
                logger.debug(f"Got info for {symbol}: {info}")
                
                if not self._is_valid_stock_data(result):
                    logger.debug(f"Symbol {symbol} failed validation checks")
                    logger.debug(f"Quote data: {info}")
                    return None
                
                return result
            return None
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return None

    async def get_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical data for a ticker symbol.
        
        Args:
            symbol: The stock symbol to query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dict containing historical data or None if the request failed
        """
        await self.rate_limiter.acquire()
        
        logger.debug(f"Getting history for {symbol} from {start_date} to {end_date}")
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Determine the period and interval
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval='1d')
            else:
                data = ticker.history(period='max', interval='1d')
            
            if data.empty:
                logger.debug(f"No historical data found for {symbol}")
                return None
                
            # Convert the data to the expected format
            result = {
                'chart': {
                    'result': [{
                        'timestamp': [int(ts.timestamp()) for ts in data.index],
                        'indicators': {
                            'quote': [{
                        'open': data['Open'].tolist(),
                                'high': data['High'].tolist(),
                                'low': data['Low'].tolist(),
                                'close': data['Close'].tolist(),
                                'volume': data['Volume'].tolist()
                            }],
                            'adjclose': [{
                                'adjclose': data['Close'].tolist()
                            }]
                        }
                    }]
                }
            }
            return result
        
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {str(e)}")
            return None

class CircuitBreaker:
    """Circuit breaker for protecting against API failures."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time in seconds before attempting to reset the circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, or half-open
        self._lock = asyncio.Lock()
        
    async def __call__(self, func, *args, **kwargs):
        """
        Execute the protected function with circuit breaker logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the protected function call
        """
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
        """
        Initialize the progress tracker.
        
        Args:
            data_dir: Directory where progress data will be stored
        """
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
        """
        Update progress for an exchange.
        
        Args:
            exchange: Name of the exchange
            total_symbols: Total number of symbols to process
            completed_symbols: Number of symbols completed
        """
        async with self._lock:
            self.progress['exchanges'][exchange] = {
                'total': total_symbols,
                'completed': completed_symbols,
                'last_update': time.time()
            }
            await self._save_progress()
                
    async def add_failed_symbol(self, exchange: str, symbol: str) -> None:
        """
        Record a failed symbol update.
        
        Args:
            exchange: Exchange the symbol belongs to
            symbol: The failed symbol
        """
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
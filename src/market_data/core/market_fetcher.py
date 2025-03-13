"""Consolidated market data fetching implementation."""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import aiofiles
from functools import lru_cache, partial
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker for API protection."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        
    def failure(self):
        """Record a failure."""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
    def success(self):
        """Record a success."""
        self.failures = 0
        self.last_failure_time = None
        
    def check(self):
        """Check if circuit is open."""
        if self.failures >= self.failure_threshold:
            if (datetime.now() - self.last_failure_time).total_seconds() < self.reset_timeout:
                raise Exception("Circuit breaker open")
            self.failures = 0

class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, rate: float, max_tokens: int):
        self.rate = rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = datetime.now()
        
    def take(self) -> bool:
        """Take a token if available."""
        now = datetime.now()
        time_passed = (now - self.last_update).total_seconds()
        self.tokens = min(
            self.max_tokens,
            self.tokens + time_passed * self.rate
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

class MarketRegistry:
    """Registry for managing market listings and symbol validation."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.markets: Dict[str, Set[str]] = {}
        self._load_markets()
    
    def _load_markets(self) -> None:
        """Load market listings from config files in both legacy and new formats."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
            
        for file_path in self.config_dir.glob("*-listed.json"):
            market_name = file_path.stem.replace("-listed", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    symbols = set()
                    
                    # Handle new standardized format
                    if isinstance(data, dict) and "symbols" in data:
                        symbols = set(data["symbols"])
                    # Handle legacy format (array of objects with Symbol/symbol)
                    elif isinstance(data, list):
                        symbols = {
                            item.get('Symbol', item.get('symbol'))
                            for item in data
                            if 'Symbol' in item or 'symbol' in item
                        }
                        
                    if symbols:
                        self.markets[market_name] = symbols
                        logger.info(f"Loaded {len(symbols)} symbols for {market_name}")
                    else:
                        logger.warning(f"No symbols found in {file_path}")
            except Exception as e:
                logger.error(f"Error loading market {market_name}: {str(e)}")
    
    def get_markets(self) -> List[str]:
        """Get list of available markets."""
        return list(self.markets.keys())
    
    def get_symbols(self, market: str) -> Set[str]:
        """Get symbols for a specific market."""
        return self.markets.get(market, set())
    
    def validate_symbol(self, market: str, symbol: str) -> bool:
        """Validate if a symbol exists in the specified market."""
        return symbol in self.markets.get(market, set())
    
    def validate_symbols(self, market: str, symbols: List[str]) -> Set[str]:
        """Validate multiple symbols for a market."""
        market_symbols = self.markets.get(market, set())
        return {symbol for symbol in symbols if symbol in market_symbols}

class MarketDataCache:
    """Efficient market data caching system."""
    
    def __init__(self, data_dir: str, ttl_days: int = 1):
        self.data_dir = Path(data_dir)
        self.ttl = timedelta(days=ttl_days)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, market: str, symbol: str, date: datetime) -> Path:
        """Get cache file path for market data."""
        return self.data_dir / market / symbol / f"{date.strftime('%Y-%m-%d')}.json"
    
    async def get(self, market: str, symbol: str, date: datetime) -> Optional[dict]:
        """Get cached market data if available and not expired."""
        cache_path = self._get_cache_path(market, symbol, date)
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        if datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime) > self.ttl:
            return None
            
        try:
            async with aiofiles.open(cache_path, 'r') as f:
                return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {str(e)}")
            return None
    
    async def set(self, market: str, symbol: str, date: datetime, data: dict):
        """Cache market data."""
        cache_path = self._get_cache_path(market, symbol, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {str(e)}")

class ConsolidatedMarketFetcher:
    """
    Optimized market data fetcher with efficient batch operations.
    
    Features:
    - Batch processing of market data requests
    - Smart caching with TTL
    - Automatic market registry management
    - Parallel processing with controlled concurrency
    - Robust error handling and logging
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "config/market",
        cache_ttl_days: int = 1,
        max_concurrent: int = 3,
        batch_size: int = 10,
        rate_limit: float = 0.5
    ):
        # Initialize components
        self.registry = MarketRegistry(config_dir)
        self.cache = MarketDataCache(data_dir, cache_ttl_days)
        
        # Configure concurrency
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Configure rate limiting
        self.token_bucket = TokenBucket(rate_limit, batch_size)
        self.circuit_breaker = CircuitBreaker(5, 300)
        
        # Configure yfinance cache
        self.cache_dir = Path(data_dir) / ".provider_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.cache_dir))
    
    async def _download_batch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data for a batch of symbols using yfinance."""
        if not symbols:
            return None
            
        async with self.semaphore:
            while not self.token_bucket.take():
                await asyncio.sleep(0.1)
                
            try:
                self.circuit_breaker.check()
                
                loop = asyncio.get_running_loop()
                download_func = partial(
                    yf.download,
                    tickers=" ".join(symbols),
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=False,
                    threads=False
                )
                
                df = await loop.run_in_executor(self.thread_pool, download_func)
                self.circuit_breaker.success()
                
                if isinstance(df.columns, pd.MultiIndex):
                    return df
                elif not df.empty:
                    df.columns = pd.MultiIndex.from_product([[symbols[0]], df.columns])
                    return df
                    
                return None
                
            except Exception as e:
                self.circuit_breaker.failure()
                logger.error(f"Error downloading batch {symbols}: {str(e)}")
                return None
            finally:
                await asyncio.sleep(1.0)  # Basic rate limiting
    
    async def fetch_market_data(
        self,
        market: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Fetch market data efficiently with caching and batch operations.
        
        Args:
            market: Market name (e.g., "nasdaq", "nyse")
            start_date: Start date for data fetching
            end_date: Optional end date (defaults to current date)
            symbols: Optional list of specific symbols (defaults to all market symbols)
        """
        if market not in self.registry.get_markets():
            raise ValueError(f"Invalid market: {market}")
            
        end_date = end_date or datetime.now()
        target_symbols = (symbols or list(self.registry.get_symbols(market)))
        valid_symbols = list(self.registry.validate_symbols(market, target_symbols))
        
        if not valid_symbols:
            logger.warning(f"No valid symbols found for market {market}")
            return
            
        logger.info(f"Fetching data for {len(valid_symbols)} symbols from {market}")
        
        for i in range(0, len(valid_symbols), self.batch_size):
            batch = valid_symbols[i:i + self.batch_size]
            try:
                df = await self._download_batch(batch, start_date, end_date)
                if df is not None and not df.empty:
                    for symbol in batch:
                        if symbol in df.columns.levels[0]:
                            symbol_data = df[symbol]
                            for date, row in symbol_data.iterrows():
                                # Skip if any essential data is missing
                                if any(pd.isna(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                                    continue
                                    
                                data = {
                                    "open": float(row['Open']),
                                    "high": float(row['High']),
                                    "low": float(row['Low']),
                                    "close": float(row['Close']),
                                    "volume": float(row['Volume']),
                                    "adjusted_close": float(row['Close'])
                                }
                                
                                await self.cache.set(
                                    market,
                                    symbol,
                                    date.to_pydatetime(),
                                    data
                                )
                            
                    logger.info(f"Processed batch {i//self.batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
    
    async def get_market_data(
        self,
        market: str,
        symbol: str,
        date: datetime
    ) -> Optional[dict]:
        """Get market data for a specific symbol and date, using cache if available."""
        if not self.registry.validate_symbol(market, symbol):
            raise ValueError(f"Invalid symbol {symbol} for market {market}")
            
        # Try cache first
        cached_data = await self.cache.get(market, symbol, date)
        if cached_data:
            return cached_data
            
        # Fetch from provider if not in cache
        try:
            df = await self._download_batch(
                [symbol],
                date,
                date + timedelta(days=1)
            )
            
            if df is not None and not df.empty and symbol in df.columns.levels[0]:
                symbol_data = df[symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    data = {
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume']),
                        "adjusted_close": float(row['Close'])
                    }
                    
                    await self.cache.set(market, symbol, date, data)
                    return data
                    
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @lru_cache(maxsize=100)
    def get_market_symbols(self, market: str) -> Set[str]:
        """Get set of symbols for a market (cached)."""
        return self.registry.get_symbols(market)
    
    def get_available_markets(self) -> List[str]:
        """Get list of available markets."""
        return self.registry.get_markets()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
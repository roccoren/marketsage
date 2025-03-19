"""Enhanced market data fetcher with advanced async processing and rate limiting."""
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
from ..core.market_fetcher import MarketRegistry
from ..core.influxdb_cache import InfluxDBMarketDataCache
from ..rate_limiting import TokenBucket, CircuitBreaker
logger = logging.getLogger(__name__)
class EnhancedMarketFetcher:
    """
    Advanced market data fetcher with enhanced async processing and reliability.
    
    Features:
    - Asynchronous batch processing with configurable chunk sizes
    - Advanced rate limiting with token bucket and circuit breaker
    - Extended market configuration support (NYSE, NASDAQ, HKEX, ASX)
    - Organized data storage with market-specific directories
    - Retry mechanisms with exponential backoff
    - Enhanced error handling and monitoring
    - Combined OHLCV and fundamental data fetching
    - InfluxDB storage for time series data
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "config/market",
        cache_ttl_days: int = 30,
        max_concurrent: int = 5,
        batch_size: int = 20,
        rate_limit: float = 0.3,
        use_influxdb: bool = True
    ):
        """
        Initialize the enhanced market data fetcher.
        
        Args:
            data_dir: Root directory for data storage
            config_dir: Directory for market configuration files
            cache_ttl_days: Days to retain cached data
            max_concurrent: Maximum concurrent operations
            batch_size: Batch size for download operations
            rate_limit: Rate limit in operations per second
            use_influxdb: Whether to use InfluxDB for data storage
        """
        # Core components
        self.registry = MarketRegistry(config_dir)
        
        # Use InfluxDB cache if enabled, otherwise use regular cache
        if use_influxdb:
            self.cache = InfluxDBMarketDataCache(data_dir, cache_ttl_days)
        else:
            from ..core.market_fetcher import MarketDataCache
            self.cache = MarketDataCache(data_dir, cache_ttl_days)
            
        # Rate limiting and concurrency
        self.token_bucket = TokenBucket(rate_limit, 30)
        self.circuit_breaker = CircuitBreaker(10, 600)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent * 2)
        
        # Performance settings
        self.retries = 3
        self.retry_delay = 1.5
        self.backoff_factor = 2.0
        
        # Disk cache for provider responses
        self.cache_dir = Path(data_dir) / ".enhanced_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Market specific paths
        self.market_paths = {}
        for market in self.registry.get_markets():
            path = Path(data_dir) / market
            path.mkdir(parents=True, exist_ok=True)
            self.market_paths[market] = path
            
        logger.info(f"Enhanced Market Fetcher initialized with batch size {batch_size}")
        
        # Cache for Ticker objects
        self._ticker_cache = {}
    async def _exponential_backoff(self, attempt: int) -> None:
        """Implement exponential backoff for retries."""
        delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
        await asyncio.sleep(delay)
    async def _download_with_retry(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data with retry mechanism and date chunking for large ranges."""
        # For very large date ranges, split into smaller chunks for more reliable downloads
        date_diff = (end_date - start_date).days
        
        # If date range is large, split into smaller chunks (90 days each)
        if date_diff > 90:
            logger.info(f"Large date range ({date_diff} days), splitting into smaller chunks for download")
            
            # Initialize an empty DataFrame to hold all results
            combined_df = None
            current_start = start_date
            
            while current_start < end_date:
                # Calculate end date for this chunk (90 days or less)
                chunk_end = min(current_start + timedelta(days=90), end_date)
                
                logger.info(f"Downloading chunk: {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                
                # Download this chunk with retry
                for attempt in range(1, self.retries + 1):
                    try:
                        chunk_df = await self._download_batch(symbols, current_start, chunk_end)
                        
                        if chunk_df is not None and not chunk_df.empty:
                            # Merge with combined results
                            if combined_df is None:
                                combined_df = chunk_df
                            else:
                                # Make sure both DataFrames have MultiIndex columns
                                if isinstance(chunk_df.columns, pd.MultiIndex) and isinstance(combined_df.columns, pd.MultiIndex):
                                    # This is a concat along index (dates)
                                    combined_df = pd.concat([combined_df, chunk_df])
                        
                        # Move to next chunk
                        break
                        
                    except Exception as e:
                        if attempt == self.retries:
                            logger.error(f"All retry attempts failed for date chunk: {str(e)}")
                        else:
                            logger.warning(f"Attempt {attempt} failed for date chunk, retrying after backoff...")
                            await self._exponential_backoff(attempt)
                
                # Move to the next chunk
                current_start = chunk_end + timedelta(days=1)
            
            # Return the combined results
            return combined_df
        
        # For smaller date ranges, download directly with retry
        for attempt in range(1, self.retries + 1):
            try:
                return await self._download_batch(symbols, start_date, end_date)
            except Exception as e:
                if attempt == self.retries:
                    logger.error(f"All retry attempts failed for symbols {symbols}: {str(e)}")
                    return None
                logger.warning(f"Attempt {attempt} failed, retrying after backoff...")
                await self._exponential_backoff(attempt)
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
            await self.token_bucket.acquire()
            
            try:
                loop = asyncio.get_running_loop()
                
                try:
                    # Try batch download first
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
                    
                    async def protected_download():
                        return await loop.run_in_executor(self.thread_pool, download_func)
                    
                    df = await self.circuit_breaker(protected_download)
                    
                    if isinstance(df.columns, pd.MultiIndex) and not df.empty:
                        return df
                except Exception as batch_error:
                    logger.warning(f"Batch download failed, falling back to individual downloads: {batch_error}")
                    
                    # Fall back to individual downloads if batch fails
                    results = {}
                    for symbol in symbols:
                        try:
                            await self.token_bucket.acquire()  # Rate limit for each symbol
                            individual_download = partial(
                                yf.download,
                                tickers=symbol,
                                start=start_date.strftime('%Y-%m-%d'),
                                end=end_date.strftime('%Y-%m-%d'),
                                group_by='ticker',
                                auto_adjust=True,
                                prepost=False,
                                threads=False,
                                period=None  # Explicitly set period to None to avoid 'max' default
                            )
                            
                            async def protected_individual_download():
                                return await loop.run_in_executor(self.thread_pool, individual_download)
                            
                            symbol_df = await self.circuit_breaker(protected_individual_download)
                            
                            if not symbol_df.empty:
                                if not isinstance(symbol_df.columns, pd.MultiIndex):
                                    symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])
                                
                                if not results:
                                    results = symbol_df
                                else:
                                    # Merge into results
                                    for col in symbol_df.columns.levels[0]:
                                        for subcol in symbol_df[col].columns:
                                            results[(col, subcol)] = symbol_df[(col, subcol)]
                        except Exception as e:
                            logger.warning(f"Could not download data for {symbol}: {str(e)}")
                    
                    if results:
                        # Convert back to DataFrame if we have any results
                        if not isinstance(results, pd.DataFrame):
                            results = pd.DataFrame(results)
                        return results
                    
                return None
            except Exception as e:
                logger.error(f"Error downloading batch {symbols}: {str(e)}")
                return None
    def _get_market_data_path(self, market: str, symbol: str) -> Path:
        """Get the path for storing market-specific data."""
        return self.market_paths[market] / symbol
    async def fetch_market_data(
        self,
        market: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Fetch market data with enhanced batch processing and error handling.
        
        Args:
            market: Market name (e.g., "nyse", "nasdaq", "hkex", "asx")
            start_date: Start date for data fetching
            end_date: Optional end date (defaults to current date)
            symbols: Optional list of specific symbols (defaults to all market symbols)
            chunk_size: Optional override for batch_size for this operation
        """
        if market not in self.registry.get_markets():
            raise ValueError(f"Invalid market: {market}")
            
        end_date = end_date or datetime.now()
        target_symbols = (symbols or list(self.registry.get_symbols(market)))
        valid_symbols = list(self.registry.validate_symbols(market, target_symbols))
        
        if not valid_symbols:
            logger.warning(f"No valid symbols found for market {market}")
            return
        batch_size = chunk_size or self.batch_size
        total_batches = (len(valid_symbols) + batch_size - 1) // batch_size
        
        logger.info(f"Fetching data for {len(valid_symbols)} symbols from {market} in {total_batches} batches")
        
        tasks = []
        for i in range(0, len(valid_symbols), batch_size):
            batch = valid_symbols[i:i + batch_size]
            tasks.append(self._process_batch(market, batch, start_date, end_date, i//batch_size + 1, total_batches))
        
        await asyncio.gather(*tasks)
    async def _process_batch(
        self,
        market: str,
        batch: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_num: int,
        total_batches: int
    ):
        """Process a batch of symbols with enhanced error handling."""
        try:
            df = await self._download_with_retry(batch, start_date, end_date)
            if df is not None and not df.empty:
                await asyncio.gather(*[
                    self._process_symbol_data(market, symbol, df, start_date, end_date)
                    for symbol in batch if symbol in df.columns.levels[0]
                ])
                
                logger.info(f"Processed batch {batch_num}/{total_batches}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {str(e)}")
    async def _process_symbol_data(
        self,
        market: str,
        symbol: str,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ):
        """Process and store data for a single symbol."""
        try:
            symbol_data = df[symbol]
            market_path = self._get_market_data_path(market, symbol)
            market_path.mkdir(parents=True, exist_ok=True)
            
            # Fetch ticker info (includes market cap and other fundamentals)
            ticker_info = await self._get_ticker_info(symbol)
            market_cap = ticker_info.get('marketCap', None)
            
            for date, row in symbol_data.iterrows():
                if any(pd.isna(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    continue
                    
                data = {
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume']),
                    "adjusted_close": float(row['Close']),
                    "date": date.strftime('%Y-%m-%d'),
                    "market": market,
                    "symbol": symbol,
                    "market_cap": market_cap
                }
                
                # Add additional fundamental data if available
                for key in ['sector', 'industry', 'fullTimeEmployees', 'longName', 'shortName']:
                    if key in ticker_info:
                        data[key] = ticker_info[key]
                
                await self.cache.set(market, symbol, date.to_pydatetime(), data)
                
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
    async def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a cached Ticker object."""
        if symbol not in self._ticker_cache:
            loop = asyncio.get_running_loop()
            
            def create_ticker():
                return yf.Ticker(symbol)
            
            self._ticker_cache[symbol] = await loop.run_in_executor(self.thread_pool, create_ticker)
        
        return self._ticker_cache[symbol]
    async def _get_ticker_info(self, symbol: str) -> Dict:
        """Fetch ticker info with caching."""
        ticker = await self._get_ticker(symbol)
        
        loop = asyncio.get_running_loop()
        
        def get_info():
            return ticker.info
        
        try:
            return await loop.run_in_executor(self.thread_pool, get_info)
        except Exception as e:
            logger.warning(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    async def get_market_data(
        self,
        market: str,
        symbol: str,
        date: datetime
    ) -> Optional[dict]:
        """Get market data for a specific symbol and date."""
        if not self.registry.validate_symbol(market, symbol):
            raise ValueError(f"Invalid symbol {symbol} for market {market}")
            
        # Try cache first
        cached_data = await self.cache.get(market, symbol, date)
        if cached_data:
            return cached_data
            
        # Fetch from provider if not in cache
        try:
            df = await self._download_with_retry(
                [symbol],
                date,
                date + timedelta(days=1)
            )
            
            if df is not None and not df.empty and symbol in df.columns.levels[0]:
                symbol_data = df[symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    
                    # Fetch ticker info (includes market cap and other fundamentals)
                    ticker_info = await self._get_ticker_info(symbol)
                    market_cap = ticker_info.get('marketCap', None)
                    
                    data = {
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume']),
                        "adjusted_close": float(row['Close']),
                        "date": date.strftime('%Y-%m-%d'),
                        "market": market,
                        "symbol": symbol,
                        "market_cap": market_cap
                    }
                    
                    # Add additional fundamental data if available
                    for key in ['sector', 'industry', 'fullTimeEmployees', 'longName', 'shortName']:
                        if key in ticker_info:
                            data[key] = ticker_info[key]
                    
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
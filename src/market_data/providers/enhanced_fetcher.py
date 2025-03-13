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
from ..core.market_fetcher import MarketRegistry, MarketDataCache
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
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "config/market",
        cache_ttl_days: int = 1,
        max_concurrent: int = 5,
        batch_size: int = 20,
        requests_per_minute: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
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
        rate_limit_ms = (60 / requests_per_minute) * 1000
        self.token_bucket = TokenBucket(rate_limit_ms, batch_size)
        self.circuit_breaker = CircuitBreaker(5, 60)
        
        # Configure retry settings
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Configure data storage
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / ".provider_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.cache_dir))
    async def _exponential_backoff(self, attempt: int) -> None:
        """Implement exponential backoff for retries."""
        delay = self.retry_delay * (2 ** (attempt - 1))
        await asyncio.sleep(delay)
    async def _download_with_retry(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data with retry mechanism."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return await self._download_batch(symbols, start_date, end_date)
            except Exception as e:
                if attempt == self.retry_attempts:
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
        return self.data_dir / market / symbol
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
                    "symbol": symbol
                }
                
                await self.cache.set(market, symbol, date.to_pydatetime(), data)
                
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
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
                    data = {
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume']),
                        "adjusted_close": float(row['Close']),
                        "date": date.strftime('%Y-%m-%d'),
                        "market": market,
                        "symbol": symbol
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
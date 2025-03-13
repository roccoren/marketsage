"""Consolidated Yahoo Finance market data provider with optimized batch operations."""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import yfinance as yf
import aiofiles

from .base import BaseMarketDataProvider
from ..models.common import (
    MarketData,
    TickerInfo,
    RealTimeQuote,
    ProviderConfig
)
from ..exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError,
    RateLimitError
)
from ..rate_limiting.circuit_breaker import CircuitBreaker
from ..rate_limiting.token_bucket import TokenBucket

logger = logging.getLogger(__name__)

class BatchingYahooProvider(BaseMarketDataProvider):
    """
    Optimized Yahoo Finance provider with batch operations and intelligent rate limiting.
    
    Features:
    - Efficient batch downloading of market data
    - Smart rate limiting with token bucket and circuit breaker
    - Concurrent operations with controlled parallelism
    - Robust error handling and retries
    - Integrated caching system
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        
        # Concurrency settings
        self.max_concurrent = min(32, config.rate_limit.burst_size)
        self.batch_size = min(10, config.rate_limit.burst_size)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent)
        
        # Rate limiting
        self.token_bucket = TokenBucket(
            rate=config.rate_limit.requests_per_second,
            max_tokens=config.rate_limit.burst_size
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=config.rate_limit.retry_after
        )
        
        # Cache configuration
        self.cache_enabled = config.cache.enabled
        self.cache_ttl = config.cache.ttl
        yf.set_tz_cache_location(
            config.cache.directory or "./data/yfinance_cache"
        )
    
    async def _wait_for_rate_limit(self):
        """Wait for rate limit token."""
        while not self.token_bucket.take():
            await asyncio.sleep(0.1)
    
    async def _validate_symbols(self, symbols: List[str]) -> Set[str]:
        """Validate multiple symbols concurrently."""
        async def validate_single(symbol: str) -> Optional[str]:
            try:
                if await self._validate_symbol_exists(symbol):
                    return symbol
            except Exception as e:
                logger.warning(f"Validation failed for {symbol}: {str(e)}")
            return None

        tasks = [validate_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {symbol for symbol in results if symbol}

    async def _download_batch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data for a batch of symbols efficiently."""
        if not symbols:
            return None
            
        async with self.semaphore:
            await self._wait_for_rate_limit()
            
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
                    # Handle single symbol case
                    df.columns = pd.MultiIndex.from_product([[symbols[0]], df.columns])
                    return df
                    
                return None
                
            except Exception as e:
                self.circuit_breaker.failure()
                if "429" in str(e) or "Too Many Requests" in str(e):
                    raise RateLimitError(f"Rate limit exceeded: {str(e)}")
                logger.error(f"Batch download error: {str(e)}")
                raise
    
    def _transform_batch_data(self, df: pd.DataFrame, symbol: str) -> List[MarketData]:
        """Transform batch DataFrame data to MarketData objects."""
        if symbol not in df.columns.levels[0]:
            return []
            
        market_data = []
        symbol_data = df[symbol]
        
        for index, row in symbol_data.iterrows():
            # Skip rows with missing essential data
            if any(pd.isna(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                continue
                
            data = MarketData(
                date=index.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                adjusted_close=float(row['Close']),  # Using adjusted data
                dividends=float(0.0),  # Not available in batch mode
                stock_splits=float(0.0)  # Not available in batch mode
            )
            market_data.append(data)
            
        return market_data
    
    async def get_history(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[MarketData]]:
        """Get historical data for multiple symbols efficiently."""
        end_date = end_date or datetime.now()
        
        # Validate symbols first
        valid_symbols = await self._validate_symbols(symbols)
        if not valid_symbols:
            raise ValidationError("No valid symbols provided")
            
        # Process symbols in batches
        results: Dict[str, List[MarketData]] = {}
        symbol_batches = [
            list(valid_symbols)[i:i + self.batch_size]
            for i in range(0, len(valid_symbols), self.batch_size)
        ]
        
        for batch in symbol_batches:
            try:
                df = await self._download_batch(batch, start_date, end_date)
                if df is not None and not df.empty:
                    for symbol in batch:
                        results[symbol] = self._transform_batch_data(df, symbol)
                        
                # Rate limiting between batches
                await asyncio.sleep(1.0)
                
            except RateLimitError:
                logger.warning("Rate limit hit, waiting before retrying...")
                await asyncio.sleep(self.circuit_breaker.reset_timeout)
                continue
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
        
        return results

    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, RealTimeQuote]:
        """Get real-time quotes for multiple symbols."""
        valid_symbols = await self._validate_symbols(symbols)
        if not valid_symbols:
            raise ValidationError("No valid symbols provided")
            
        quotes: Dict[str, RealTimeQuote] = {}
        
        async def fetch_quote(symbol: str) -> Optional[RealTimeQuote]:
            async with self.semaphore:
                await self._wait_for_rate_limit()
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.fast_info
                    
                    return RealTimeQuote(
                        symbol=symbol,
                        price=info.last_price,
                        change=info.last_price - info.previous_close,
                        change_percent=((info.last_price - info.previous_close) / info.previous_close) * 100,
                        volume=info.last_volume,
                        timestamp=datetime.now(),
                        bid=None,  # Not reliably available
                        ask=None,  # Not reliably available
                        last_trade_time=info.last_trade_time
                    )
                except Exception as e:
                    logger.error(f"Error fetching quote for {symbol}: {str(e)}")
                    return None
        
        tasks = [fetch_quote(symbol) for symbol in valid_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, quote in zip(valid_symbols, results):
            if isinstance(quote, RealTimeQuote):
                quotes[symbol] = quote
        
        return quotes

    async def get_ticker_info(self, symbols: List[str]) -> Dict[str, TickerInfo]:
        """Get ticker information for multiple symbols."""
        valid_symbols = await self._validate_symbols(symbols)
        if not valid_symbols:
            raise ValidationError("No valid symbols provided")
            
        info: Dict[str, TickerInfo] = {}
        
        async def fetch_info(symbol: str) -> Optional[TickerInfo]:
            async with self.semaphore:
                await self._wait_for_rate_limit()
                try:
                    ticker = yf.Ticker(symbol)
                    ticker_info = ticker.info
                    
                    return TickerInfo(
                        symbol=symbol,
                        name=ticker_info.get('longName'),
                        exchange=ticker_info.get('exchange'),
                        currency=ticker_info.get('currency'),
                        market_cap=ticker_info.get('marketCap'),
                        sector=ticker_info.get('sector'),
                        industry=ticker_info.get('industry')
                    )
                except Exception as e:
                    logger.error(f"Error fetching info for {symbol}: {str(e)}")
                    return None
        
        tasks = [fetch_info(symbol) for symbol in valid_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, ticker_info in zip(valid_symbols, results):
            if isinstance(ticker_info, TickerInfo):
                info[symbol] = ticker_info
        
        return info

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
"""Optimized market data fetcher with efficient batch operations and caching."""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import aiofiles
from functools import lru_cache

from .providers.consolidated_yahoo import BatchingYahooProvider
from .models.common import (
    MarketData,
    ProviderConfig,
    RateLimitConfig,
    CacheConfig
)
from .exceptions import ValidationError

logger = logging.getLogger(__name__)

class MarketRegistry:
    """Registry for managing market listings and symbol validation."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.markets: Dict[str, Set[str]] = {}
        self._load_markets()
    
    def _load_markets(self) -> None:
        """Load market listings from config files."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
            
        for file_path in self.config_dir.glob("*-listed.json"):
            market_name = file_path.stem.replace("-listed", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    symbols = {
                        item.get('Symbol', item.get('symbol'))
                        for item in data
                        if 'Symbol' in item or 'symbol' in item
                    }
                    if symbols:
                        self.markets[market_name] = symbols
                        logger.info(f"Loaded {len(symbols)} symbols for {market_name}")
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
    
    async def get(self, market: str, symbol: str, date: datetime) -> Optional[MarketData]:
        """Get cached market data if available and not expired."""
        cache_path = self._get_cache_path(market, symbol, date)
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        if datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime) > self.ttl:
            return None
            
        try:
            async with aiofiles.open(cache_path, 'r') as f:
                data = json.loads(await f.read())
                return MarketData(
                    date=date,
                    **data
                )
        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {str(e)}")
            return None
    
    async def set(self, market: str, symbol: str, data: MarketData):
        """Cache market data."""
        cache_path = self._get_cache_path(market, symbol, data.date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data_dict = {
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "close": float(data.close),
                "volume": float(data.volume),
                "adjusted_close": float(data.adjusted_close),
                "dividends": float(getattr(data, "dividends", 0) or 0),
                "stock_splits": float(getattr(data, "stock_splits", 0) or 0)
            }
            
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(data_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {str(e)}")

class MarketDataFetcher:
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
        batch_size: int = 10
    ):
        # Initialize components
        self.registry = MarketRegistry(config_dir)
        self.cache = MarketDataCache(data_dir, cache_ttl_days)
        
        # Configure provider
        provider_config = ProviderConfig(
            name="yahoo",
            rate_limit=RateLimitConfig(
                requests_per_second=0.5,  # 2 requests per second max
                burst_size=batch_size,
                retry_after=300  # 5 minutes cooldown
            ),
            cache=CacheConfig(
                enabled=True,
                ttl=86400,  # 1 day cache
                directory=str(Path(data_dir) / ".provider_cache")
            )
        )
        
        self.provider = BatchingYahooProvider(provider_config)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
    
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
            raise ValidationError(f"Invalid market: {market}")
            
        end_date = end_date or datetime.now()
        target_symbols = (symbols or list(self.registry.get_symbols(market)))
        valid_symbols = list(self.registry.validate_symbols(market, target_symbols))
        
        if not valid_symbols:
            logger.warning(f"No valid symbols found for market {market}")
            return
            
        logger.info(f"Fetching data for {len(valid_symbols)} symbols from {market}")
        
        # Process symbols in optimally-sized batches
        for i in range(0, len(valid_symbols), self.batch_size):
            batch = valid_symbols[i:i + self.batch_size]
            try:
                # Fetch batch data
                results = await self.provider.get_history(
                    batch,
                    start_date,
                    end_date
                )
                
                # Cache results
                for symbol, data_points in results.items():
                    for data in data_points:
                        await self.cache.set(market, symbol, data)
                        
                logger.info(f"Processed batch {i//self.batch_size + 1} for {market}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
    
    async def get_market_data(
        self,
        market: str,
        symbol: str,
        date: datetime
    ) -> Optional[MarketData]:
        """Get market data for a specific symbol and date, using cache if available."""
        if not self.registry.validate_symbol(market, symbol):
            raise ValidationError(f"Invalid symbol {symbol} for market {market}")
            
        # Try cache first
        cached_data = await self.cache.get(market, symbol, date)
        if cached_data:
            return cached_data
            
        # Fetch from provider if not in cache
        try:
            results = await self.provider.get_history(
                [symbol],
                date,
                date + timedelta(days=1)
            )
            
            if symbol in results and results[symbol]:
                data = results[symbol][0]
                await self.cache.set(market, symbol, data)
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
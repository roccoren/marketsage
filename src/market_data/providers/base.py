"""Base class for market data providers."""
import asyncio
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..interfaces import MarketDataProvider, Cache
from ..models.common import (
    MarketData,
    HistoricalData,
    TickerInfo,
    RequestParams,
    ProviderConfig
)
from ..exceptions import (
    ValidationError,
    InvalidSymbolError,
    InvalidDateRangeError,
    DataNotFoundError
)
from ..rate_limiting.token_bucket import TokenBucket
from ..rate_limiting.circuit_breaker import CircuitBreaker

class BaseMarketDataProvider(MarketDataProvider):
    """
    Base implementation of market data provider with common functionality.
    
    Features:
    - Rate limiting
    - Circuit breaker protection
    - Request validation
    - Response transformation
    - Optional caching
    """
    
    def __init__(
        self,
        config: ProviderConfig,
        cache: Optional[Cache] = None,
    ):
        """
        Initialize provider with configuration.
        
        Args:
            config: Provider configuration
            cache: Optional cache implementation
        """
        self.config = config
        self.cache = cache
        
        # Set up rate limiting
        self.rate_limiter = TokenBucket(config.rate_limit)
        
        # Set up circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300.0  # 5 minutes
        )
        
        self._lock = asyncio.Lock()

    @abstractmethod
    async def _fetch_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """
        Fetch historical data from provider.
        
        This method must be implemented by provider classes.
        """
        pass
        
    @abstractmethod
    async def _fetch_ticker_info(self, symbol: str) -> TickerInfo:
        """
        Fetch ticker information from provider.
        
        This method must be implemented by provider classes.
        """
        pass

    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate and normalize symbol.
        
        Args:
            symbol: Raw symbol string
            
        Returns:
            Normalized symbol string
            
        Raises:
            InvalidSymbolError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError("Symbol must be a non-empty string")
            
        # Remove common prefixes/suffixes
        symbol = symbol.strip().upper()
        if symbol.startswith('$'):
            symbol = symbol[1:]
            
        return symbol

    def _validate_dates(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> tuple[datetime, datetime]:
        """
        Validate and normalize date range.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Tuple of (start_date, end_date)
            
        Raises:
            InvalidDateRangeError: If date range is invalid
        """
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            start_date = end_date - timedelta(days=365)
            
        if start_date > end_date:
            raise InvalidDateRangeError("Start date must be before end date")
            
        if start_date > datetime.now():
            raise InvalidDateRangeError("Start date cannot be in the future")
            
        return start_date, end_date

    async def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for historical data."""
        return f"{self.config.name}:history:{symbol}:{start_date.date()}:{end_date.date()}"

    async def get_history(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Fetch historical market data for a single symbol.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary containing historical data
            
        Raises:
            Various exceptions defined in exceptions.py
        """
        # Validate inputs
        symbol = self._validate_symbol(symbol)
        start_date, end_date = self._validate_dates(start_date, end_date)
        
        # Check cache first
        if self.cache is not None:
            cache_key = await self._get_cache_key(symbol, start_date, end_date)
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Acquire rate limit token
        await self.rate_limiter.acquire()
        
        # Fetch data with circuit breaker protection
        try:
            data = await self.circuit_breaker.call(
                self._fetch_history,
                symbol,
                start_date,
                end_date
            )
        except Exception as e:
            raise DataNotFoundError(f"Failed to fetch data for {symbol}: {str(e)}")
            
        if not data:
            raise DataNotFoundError(f"No data available for {symbol}")
            
        # Transform to common format
        result = HistoricalData(
            symbol=symbol,
            data=data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Cache the result
        if self.cache is not None:
            await self.cache.set(
                await self._get_cache_key(symbol, start_date, end_date),
                result
            )
            
        return result

    async def get_history_batch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch historical market data for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary mapping symbols to their historical data
        """
        results = {}
        async with asyncio.TaskGroup() as group:
            for symbol in symbols:
                group.create_task(
                    self._fetch_single(symbol, start_date, end_date, results)
                )
        return results
        
    async def _fetch_single(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        results: Dict[str, Any]
    ) -> None:
        """Helper method for batch fetching."""
        try:
            data = await self.get_history(symbol, start_date, end_date)
            results[symbol] = data
        except Exception as e:
            results[symbol] = None

    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch general information about a ticker.
        
        Args:
            symbol: The symbol to fetch info for
            
        Returns:
            Dictionary containing ticker information
        """
        symbol = self._validate_symbol(symbol)
        
        # Check cache first
        if self.cache is not None:
            cache_key = f"{self.config.name}:info:{symbol}"
            cached_info = await self.cache.get(cache_key)
            if cached_info is not None:
                return cached_info
        
        # Acquire rate limit token
        await self.rate_limiter.acquire()
        
        # Fetch info with circuit breaker protection
        try:
            info = await self.circuit_breaker.call(
                self._fetch_ticker_info,
                symbol
            )
        except Exception as e:
            raise DataNotFoundError(f"Failed to fetch info for {symbol}: {str(e)}")
            
        if not info:
            raise DataNotFoundError(f"No info available for {symbol}")
            
        # Cache the result
        if self.cache is not None:
            await self.cache.set(
                f"{self.config.name}:info:{symbol}",
                info,
                ttl=3600  # Cache ticker info for 1 hour
            )
            
        return info
"""Market data provider interfaces."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_history(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Fetch historical market data for a single symbol."""
        pass
        
    @abstractmethod
    async def get_history_batch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch historical market data for multiple symbols."""
        pass
        
    @abstractmethod
    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch general information about a ticker."""
        pass

class Cache(ABC):
    """Abstract base class for caching implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache."""
        pass
        
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in cache with optional TTL in seconds."""
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a value from cache."""
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass

class RateLimiter(ABC):
    """Abstract base class for rate limiting implementations."""
    
    @abstractmethod
    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        pass
        
    @abstractmethod
    def get_delay(self) -> float:
        """Get current delay in seconds before next token."""
        pass
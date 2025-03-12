"""Interfaces for market data package."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class Cache(ABC):
    """Interface for caching implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
        
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

class MarketDataProvider(ABC):
    """Interface for market data providers."""
    
    @abstractmethod
    async def get_history(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get historical market data for a symbol."""
        pass
        
    @abstractmethod
    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a ticker/symbol."""
        pass
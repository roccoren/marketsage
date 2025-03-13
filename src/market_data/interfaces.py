"""Interfaces for market data package."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from .models.common import RealTimeQuote, MarketSource

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

    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> RealTimeQuote:
        """
        Get real-time quote data for a symbol.
        
        Args:
            symbol: The symbol to fetch quote for
            
        Returns:
            RealTimeQuote object with current market data
            
        Raises:
            DataNotFoundError: If quote cannot be retrieved
        """
        pass

    @abstractmethod
    async def get_real_time_quotes_batch(self, symbols: List[str]) -> Dict[str, RealTimeQuote]:
        """
        Get real-time quotes for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch quotes for
            
        Returns:
            Dictionary mapping symbols to their real-time quotes
            
        Note: Failed quotes will be omitted from the results
        """
        pass

class MarketSymbolLoader(ABC):
    """Interface for loading market symbols."""

    @abstractmethod
    async def load_symbols(self, market: MarketSource) -> List[str]:
        """
        Load symbols for a specific market.

        Args:
            market: Market source to load symbols for

        Returns:
            List of symbol strings
            
        Raises:
            FileNotFoundError: If symbols file is missing
            ValueError: If file contains invalid data
        """
        pass

    @abstractmethod
    async def get_market_metadata(self, market: MarketSource) -> Dict[str, Any]:
        """
        Get metadata for a specific market.
        
        Args:
            market: Market source to get metadata for
            
        Returns:
            Dictionary containing market metadata
        """
        pass
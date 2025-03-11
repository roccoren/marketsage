"""Market data package for fetching and managing financial market data."""

from typing import Union, Type

# Import commonly used components
from .interfaces import MarketDataProvider, Cache
from .models.common import (
    MarketData,
    TickerInfo,
    HistoricalData,
    RequestParams,
    ProviderConfig,
    CacheConfig,
    RateLimitConfig
)
from .exceptions import (
    MarketDataError,
    ProviderError,
    RateLimitError,
    CircuitBreakerError,
    ValidationError,
    CacheError,
    ConnectionError,
    DataNotFoundError,
    InvalidSymbolError,
    InvalidDateRangeError,
    ConfigurationError
)
from .providers.yahoo import YahooFinanceProvider
from .cache.disk import DiskCache

# Version info
__version__ = '0.1.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Factory function to create provider instances
def create_provider(
    provider_type: Union[str, Type[MarketDataProvider]],
    config: ProviderConfig
) -> MarketDataProvider:
    """
    Create a market data provider instance.
    
    Args:
        provider_type: Either a provider class or string name ('yahoo')
        config: Provider configuration
        
    Returns:
        Configured provider instance
        
    Raises:
        ValueError: If provider_type is invalid
    """
    if isinstance(provider_type, str):
        if provider_type.lower() == 'yahoo':
            return YahooFinanceProvider(config)
        raise ValueError(f"Unknown provider type: {provider_type}")
        
    if isinstance(provider_type, type) and issubclass(provider_type, MarketDataProvider):
        return provider_type(config)
        
    raise ValueError("provider_type must be a string or MarketDataProvider subclass")

# Default export list
__all__ = [
    # Interfaces
    'MarketDataProvider',
    'Cache',
    
    # Models
    'MarketData',
    'TickerInfo',
    'HistoricalData',
    'RequestParams',
    'ProviderConfig',
    'CacheConfig',
    'RateLimitConfig',
    
    # Exceptions
    'MarketDataError',
    'ProviderError',
    'RateLimitError',
    'CircuitBreakerError',
    'ValidationError',
    'CacheError',
    'ConnectionError',
    'DataNotFoundError',
    'InvalidSymbolError',
    'InvalidDateRangeError',
    'ConfigurationError',
    
    # Providers
    'YahooFinanceProvider',
    
    # Cache implementations
    'DiskCache',
    
    # Factory functions
    'create_provider',
]
"""Market data package initialization."""
from .core.market_fetcher import ConsolidatedMarketFetcher
from .providers.yahoo import YahooFinanceProvider
from .models.common import (
    MarketData,
    TickerInfo,
    HistoricalData,
    RequestParams,
    CacheConfig,
    RateLimitConfig,
    ProviderConfig
)
from .exceptions import (
    MarketDataError,
    ProviderError,
    RateLimitError,
    CircuitBreakerError,
    ValidationError,
    CacheError,
    ConnectionError,
    DataNotFoundError
)

__version__ = "0.1.0"

# Set default exports for cleaner imports
__all__ = [
    'ConsolidatedMarketFetcher',  # Primary interface
    'YahooFinanceProvider',       # Provider-level access
    'MarketData',
    'TickerInfo',
    'HistoricalData',
    'RequestParams',
    'CacheConfig',
    'RateLimitConfig',
    'ProviderConfig',
    'MarketDataError',
    'ProviderError',
    'RateLimitError',
    'CircuitBreakerError',
    'ValidationError',
    'CacheError',
    'ConnectionError',
    'DataNotFoundError'
]
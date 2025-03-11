"""Custom exceptions for market data package."""

class MarketDataError(Exception):
    """Base exception for all market data related errors."""
    pass

class ProviderError(MarketDataError):
    """Base exception for provider-specific errors."""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limits are exceeded."""
    pass

class CircuitBreakerError(ProviderError):
    """Raised when circuit breaker is open."""
    def __init__(self, failure_count: int, message: str = "Circuit breaker is open"):
        self.failure_count = failure_count
        super().__init__(message)

class ValidationError(MarketDataError):
    """Raised when data validation fails."""
    pass

class CacheError(MarketDataError):
    """Base exception for cache-related errors."""
    pass

class ConnectionError(ProviderError):
    """Raised when connection to data provider fails."""
    pass

class DataNotFoundError(ProviderError):
    """Raised when requested data is not available."""
    pass

class InvalidSymbolError(ValidationError):
    """Raised when symbol format is invalid."""
    pass

class InvalidDateRangeError(ValidationError):
    """Raised when date range is invalid."""
    def __init__(self, message: str = "Invalid date range provided"):
        super().__init__(message)

class ConfigurationError(MarketDataError):
    """Raised when there are configuration issues."""
    pass
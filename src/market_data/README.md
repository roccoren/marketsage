# Market Data Package

A robust and efficient package for fetching and managing market data from various providers. Currently supports Yahoo Finance with an extensible architecture for adding additional providers.

## Features

- **Efficient Data Fetching**
  - Asynchronous operations
  - Connection pooling
  - Request batching
  - Automatic retries

- **Rate Limiting**
  - Token bucket algorithm
  - Configurable rates and burst sizes
  - Automatic request queuing

- **Circuit Breaking**
  - Fault tolerance
  - Automatic service recovery
  - Failure tracking
  - Configurable thresholds

- **Caching**
  - Disk-based caching
  - Configurable TTL
  - Automatic compression
  - Thread-safe operations

- **Error Handling**
  - Custom exception hierarchy
  - Detailed error messages
  - Request validation
  - Response validation

## Installation

```bash
# Install from your project root
pip install -e src/market_data
```

## Quick Start

```python
import asyncio
from datetime import datetime
from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import ProviderConfig, RateLimitConfig, CacheConfig

async def main():
    # Configure the provider
    config = ProviderConfig(
        name="yahoo",
        rate_limit=RateLimitConfig(
            requests_per_second=2.0,
            burst_size=5
        ),
        cache=CacheConfig(
            enabled=True,
            ttl=3600,  # 1 hour
            directory="./cache"
        )
    )
    
    # Create provider instance
    provider = YahooFinanceProvider(config)
    
    # Fetch historical data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    try:
        data = await provider.get_history("AAPL", start_date, end_date)
        print(f"Retrieved {len(data.data)} data points")
        
        # Fetch multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        batch_data = await provider.get_history_batch(symbols, start_date, end_date)
        
        for symbol, result in batch_data.items():
            if result:
                print(f"{symbol}: {len(result.data)} data points")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Provider Configuration

```python
ProviderConfig(
    name="yahoo",                  # Provider name
    api_key="your_key",           # Optional API key
    api_secret="your_secret",     # Optional API secret
    base_url="api.example.com",   # Optional base URL
    timeout=30,                   # Request timeout in seconds
    max_retries=3,               # Maximum retry attempts
    cache=CacheConfig(...),      # Cache configuration
    rate_limit=RateLimitConfig(...) # Rate limit configuration
)
```

### Rate Limiting Configuration

```python
RateLimitConfig(
    requests_per_second=2.0,  # Request rate
    burst_size=5,            # Maximum burst size
    retry_after=60          # Seconds to wait after hitting limit
)
```

### Cache Configuration

```python
CacheConfig(
    enabled=True,           # Enable/disable caching
    ttl=3600,              # Default TTL in seconds
    max_size=1000000,      # Optional max cache size in bytes
    directory="./cache"    # Cache directory path
)
```

## Error Handling

The package provides specific exceptions for different error cases:

```python
from market_data.exceptions import (
    MarketDataError,        # Base exception
    ProviderError,          # Provider-specific errors
    RateLimitError,         # Rate limit exceeded
    CircuitBreakerError,    # Circuit breaker is open
    ValidationError,        # Invalid input data
    CacheError,            # Cache operation errors
    ConnectionError,       # Network/connection issues
    DataNotFoundError      # Requested data not available
)

try:
    data = await provider.get_history(...)
except RateLimitError:
    print("Rate limit exceeded")
except CircuitBreakerError:
    print("Service is currently unavailable")
except ValidationError as e:
    print(f"Invalid input: {str(e)}")
except DataNotFoundError:
    print("No data available")
```

## Adding New Providers

To add a new provider:

1. Create a new class that extends `BaseMarketDataProvider`
2. Implement the required abstract methods:
   - `_fetch_history`
   - `_fetch_ticker_info`
3. Optionally override other methods for custom behavior

Example:

```python
from market_data.providers.base import BaseMarketDataProvider
from market_data.models.common import MarketData, TickerInfo

class CustomProvider(BaseMarketDataProvider):
    async def _fetch_history(self, symbol, start_date, end_date):
        # Implement data fetching logic
        pass
        
    async def _fetch_ticker_info(self, symbol):
        # Implement info fetching logic
        pass
```

## Running Tests

```bash
# From project root
pytest tests/market_data/
```

## Contributing

1. Write tests for new functionality
2. Ensure all tests pass
3. Follow existing code style
4. Update documentation as needed
5. Submit pull request

## License

MIT License
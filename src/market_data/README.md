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
"""Example demonstrating usage of the market data package with Yahoo Finance."""
import os
from datetime import datetime, timedelta
import asyncio
from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import (
    ProviderConfig,
    CacheConfig,
    RateLimitConfig
)
from market_data.exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError
)

async def main():
    # Ensure cache directory exists
    cache_dir = "./data/cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize provider configuration
    config = ProviderConfig(
        name="yahoo",
        cache=CacheConfig(
            enabled=True,
            directory=cache_dir,
            ttl=24 * 3600  # 1 day cache
        ),
        rate_limit=RateLimitConfig(
            requests_per_second=0.2,  # Conservative rate limiting
            burst_size=1,
            retry_after=5  # Wait 5 seconds after hitting rate limit
        )
    )
    
    # Initialize the provider
    provider = YahooFinanceProvider(config)

    # Define time range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last week of data
    
    try:
        # First try to get ticker info
        symbol = "AAPL"
        info = await provider.get_ticker_info(symbol)
        print(f"Found ticker: {info.name} ({info.exchange})")

        # Then fetch historical data
        result = await provider.get_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if result and result.data:
            print(f"\nData summary for {symbol}:")
            print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
            print(f"Number of data points: {len(result.data)}")
            
            # Show latest trading day
            latest = result.data[-1]
            print(f"\nLatest trading day ({latest.date.strftime('%Y-%m-%d')}):")
            print(f"Open:     ${latest.open:.2f}")
            print(f"Close:    ${latest.close:.2f}")
            print(f"Volume:   {latest.volume:,}")
            
    except ValidationError as e:
        print(f"Invalid symbol: {str(e)}")
    except DataNotFoundError as e:
        print(f"No data found: {str(e)}")
    except ConnectionError as e:
        print(f"Connection error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        # Cleanup resources
        if hasattr(provider, '__del__'):
            provider.__del__()

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
    # First validate the symbol
    info = await provider.get_ticker_info(symbol)
    
    # Then get historical data
    data = await provider.get_history(symbol, start_date, end_date)
except RateLimitError:
    print("Rate limit exceeded")
except CircuitBreakerError:
    print("Service is currently unavailable")
except ValidationError as e:
    print(f"Invalid input: {str(e)}")
except DataNotFoundError:
    print("No data available")
except ConnectionError as e:
    print(f"Connection failed: {str(e)}")
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
# Market Data Package

A robust and efficient package for fetching and managing market data from various providers. Currently supports Yahoo Finance with an extensible architecture for adding additional providers.

## Features

- **High-Level Market Data Fetching**
  - Consolidated market fetcher with registry support
  - Batch processing and backfilling capabilities
  - Market symbol validation and management
  - Efficient data storage and retrieval

- **Rate Limiting & Protection**
  - Token bucket algorithm
  - Circuit breaker pattern
  - Configurable rates and burst sizes
  - Automatic request queuing

- **Smart Caching**
  - Disk-based caching with TTL
  - Automatic compression
  - Thread-safe operations
  - Configurable storage options

- **Error Handling**
  - Custom exception hierarchy
  - Detailed error messages
  - Request/response validation
  - Automatic retries

## Quick Start

### Using the Consolidated Market Fetcher

```python
from datetime import datetime
import asyncio
from market_data.core.market_fetcher import ConsolidatedMarketFetcher

async def main():
    # Initialize the fetcher
    fetcher = ConsolidatedMarketFetcher(
        data_dir="data",
        config_dir="config/market",
        cache_ttl_days=1,
        max_concurrent=3,
        batch_size=10,
        rate_limit=0.5  # 2 requests per second max
    )
    
    # Get available markets
    markets = fetcher.get_available_markets()
    print(f"Available markets: {markets}")
    
    # Fetch data for a specific symbol
    market = "nasdaq"
    symbol = "AAPL"
    date = datetime.now()
    
    data = await fetcher.get_market_data(market, symbol, date)
    if data:
        print(f"Latest data for {symbol}:")
        print(f"Open: ${data['open']:.2f}")
        print(f"Close: ${data['close']:.2f}")
        print(f"Volume: {data['volume']:,}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Low-Level Provider API

```python
from datetime import datetime, timedelta
import asyncio
from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import (
    ProviderConfig,
    CacheConfig,
    RateLimitConfig
)

async def main():
    config = ProviderConfig(
        name="yahoo",
        cache=CacheConfig(
            enabled=True,
            directory="./cache",
            ttl=24 * 3600  # 1 day cache
        ),
        rate_limit=RateLimitConfig(
            requests_per_second=0.2,
            burst_size=1,
            retry_after=5
        )
    )
    
    provider = YahooFinanceProvider(config)
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    result = await provider.get_history(symbol, start_date, end_date)
    if result and result.data:
        latest = result.data[-1]
        print(f"Latest data for {symbol}:")
        print(f"Open: ${latest.open:.2f}")
        print(f"Close: ${latest.close:.2f}")
        print(f"Volume: {latest.volume:,}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Consolidated Market Fetcher Configuration

```python
ConsolidatedMarketFetcher(
    data_dir="data",              # Base directory for data storage
    config_dir="config/market",   # Directory containing market listings
    cache_ttl_days=1,            # Cache time-to-live in days
    max_concurrent=3,            # Maximum concurrent requests
    batch_size=10,              # Symbols to process in each batch
    rate_limit=0.5              # Requests per second limit
)
```

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
    data = await fetcher.get_market_data(market, symbol, date)
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
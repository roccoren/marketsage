# Enhanced Market Data Fetcher

A flexible and robust system for fetching and managing stock market data across multiple exchanges. This enhanced version builds upon the base MarketDataFetcher with additional features for handling multiple exchanges, data backfilling, and gap detection.

## Features

- **Multi-Exchange Support**: Seamlessly work with data from multiple stock exchanges (NASDAQ, NYSE, HKEX, ASX, etc.)
- **Automatic Backfilling**: Detect and fill gaps in historical data
- **Efficient Data Storage**: Organized directory structure for easy data access
- **Rate Limiting**: Built-in rate limiting and circuit breaker protection
- **Extensible Design**: Easy to add new market sources and data providers

## Directory Structure

```
data/
├── config/
│   ├── nasdaq-listed.json
│   ├── nyse-listed.json
│   └── ...
├── nasdaq/
│   ├── AAPL/
│   │   ├── 2024-01-01.json
│   │   └── ...
│   └── ...
└── nyse/
    ├── IBM/
    │   ├── 2024-01-01.json
    │   └── ...
    └── ...
```

## Usage

### Command Line Interface

The package provides a convenient command-line tool `fetch_market_data.py` for fetching market data:

```bash
# Fetch AAPL data from NASDAQ for last month
python fetch_market_data.py --market nasdaq --symbol AAPL --start-date 2024-02-01

# Fetch all NASDAQ symbols for 2023
python fetch_market_data.py --market nasdaq --start-date 2023-01-01 --end-date 2023-12-31

# Backfill specific symbol with custom cache directory
python fetch_market_data.py --market nyse --symbol IBM --start-date 2020-01-01 --cache-dir /path/to/cache
```

#### Command Line Options

- `--market`: Market name (e.g., nasdaq, nyse) [required]
- `--symbol`: Specific symbol to fetch (optional)
- `--start-date`: Start date in YYYY-MM-DD format (default: 2020-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: today)
- `--data-dir`: Directory for storing market data (default: data)
- `--cache-dir`: Directory for cache storage (default: .cache)
- `--config-dir`: Directory containing market listings (default: data/config)
- `--verbose`: Enable verbose logging

### Programmatic Usage

```python
from datetime import datetime
from enhanced_market_fetcher import EnhancedMarketDataFetcher

# Initialize fetcher
fetcher = EnhancedMarketDataFetcher()

# Fetch recent data for entire market
start_date = datetime(2024, 1, 1)
await fetcher.fetch_market_data("nasdaq", start_date)

# Backfill specific symbol
await fetcher.backfill_data("nasdaq", "AAPL", start_date)
```

### Market Registry

The `MarketRegistry` class manages market listings and symbol validation:

```python
# Get available markets
markets = fetcher.registry.get_markets()

# Get symbols for a market
nasdaq_symbols = fetcher.registry.get_symbols("nasdaq")

# Validate symbol
is_valid = fetcher.registry.validate_symbol("nasdaq", "AAPL")
```

### Data Backfilling

Automatically detect and fill gaps in historical data:

```python
# Backfill with specific date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)
await fetcher.backfill_data("nasdaq", "AAPL", start_date, end_date)

# Backfill to current date
await fetcher.backfill_data("nasdaq", "AAPL", start_date)
```

## Configuration

### Market Listings

Market listings are stored in JSON files under `data/config/` with the naming pattern `{market}-listed.json`. Each file should contain an array of objects with at least a `symbol` field:

```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology"
  }
]
```

### Provider Configuration

The system uses the Yahoo Finance provider by default with optimized settings:

```python
config = ProviderConfig(
    name="yahoo",
    rate_limit=RateLimitConfig(
        requests_per_second=2.0,
        burst_size=5,
        retry_after=60
    ),
    cache=CacheConfig(
        enabled=True,
        ttl=86400,  # 24 hours
        directory=".cache"
    )
)
```

## Adding New Markets

To add support for a new market:

1. Create a new market listing file in `data/config/{market}-listed.json`
2. The registry will automatically detect and load the new market
3. Use the fetcher with the new market name:
   ```python
   await fetcher.fetch_market_data("new_market", start_date)
   ```

## Error Handling

The system includes comprehensive error handling for:

- Rate limiting (`RateLimitError`)
- Service availability (`CircuitBreakerError`)
- Data validation (`ValidationError`)
- Invalid markets/symbols (`ValueError`)

## Examples

See `example_usage.py` for comprehensive usage examples including:

- Basic market data fetching
- Data backfilling
- Working with multiple markets
- Gap detection and filling
- Error handling

## Testing

Run the test suite:

```bash
pytest test_enhanced_fetcher.py
```

The tests cover:

- Market registry functionality
- Data backfilling
- Gap detection
- Error handling
- Multiple market support

## Dependencies

- Python 3.8+
- market_data package (internal)
- asyncio
- pytest (for testing)

## Contributing

1. Follow established patterns in `market_data_fetcher.py`
2. Add tests for new functionality
3. Update documentation for significant changes
4. Ensure all tests pass before submitting changes
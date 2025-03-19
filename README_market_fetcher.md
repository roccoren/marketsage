# Enhanced Market Data Fetcher with InfluxDB Storage

This tool fetches stock market data for specified symbols and stores it directly in InfluxDB for efficient time series analysis.

## Features

- Direct symbol fetching without market constraints
- Loads symbols from text or JSON files
- Fetches stock data with proper rate limiting and error handling
- Includes market cap and fundamental data in the results
- Stores data in InfluxDB for efficient time series analysis
- Detailed logging and error reporting
- Graceful shutdown with Ctrl+C
- Configurable timeouts for long-running operations

## Usage

### Basic Usage with Symbol List

Fetch data for all symbols in a file and store in InfluxDB:

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json
```

or 

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.txt
```

### Direct Symbol Input

```bash
python -m src.market_system.fetch_market_data --symbols "AAPL,MSFT,GOOG,AMZN"
```

### Set Operation Timeout

Automatically terminate the operation after a set time (e.g., 5 minutes):

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --timeout 300
```

### Specify Date Range

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --start-date 2022-01-01 --end-date 2023-01-01
```

### Specify Market (Optional)

If you want to only fetch symbols from a specific market (not required):

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --market nasdaq
```

### Performance Tuning

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --max-concurrent 10 --batch-size 30 --rate-limit 0.8
```

### Force InfluxDB Storage

Force using InfluxDB even if configuration specifies file storage:

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --force-influxdb
```

### Verbose Mode

For detailed logging and debugging information:

```bash
python -m src.market_system.fetch_market_data --symbols-file sample_symbols.json --verbose
```

## Symbol File Formats

### JSON Format

```json
{
    "symbols": [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN"
    ]
}
```

### Text Format

```
AAPL
MSFT
GOOG
AMZN
```

## InfluxDB Storage

All data is stored in InfluxDB with the following structure:

- **Measurement**: `stock_data`
- **Tags**:
  - `symbol`: The stock symbol (e.g., "AAPL")
  - `market`: The market identifier (e.g., "nasdaq", "nyse", "hkex", "asx" or "custom")
- **Fields**:
  - `open`: Opening price
  - `high`: High price
  - `low`: Low price
  - `close`: Closing price
  - `volume`: Trading volume
  - `adjusted_close`: Adjusted closing price
  - Additional fields if available (market_cap, sector, etc.)

## Querying Data

### Basic Flux Query Example

```flux
from(bucket: "stocks")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "stock_data")
  |> filter(fn: (r) => r.symbol == "AAPL")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
```

## Configuration

InfluxDB connection parameters can be configured in `config/storage_config.json`:

```json
{
  "storage_type": "influxdb",
  "influxdb": {
    "url": "http://localhost:8086",
    "token": "your-token-here",
    "org": "your-org",
    "bucket": "stocks"
  }
}
```

## Troubleshooting

### Common Issues

1. **"No data found for symbol X"**:
   - The symbol might be delisted or not available on Yahoo Finance
   - Try using a different symbol format (e.g., add '.HK' for Hong Kong stocks)
   - Ensure the date range is valid for the symbol

2. **"YFTzMissingError: possibly delisted; no timezone found"**:
   - The symbol may be delisted or incorrectly formatted
   - Some symbols have changed (e.g., 'FB' is now 'META')

3. **"Failed to connect to InfluxDB"**:
   - Check that InfluxDB is running at the configured URL
   - Verify that your authentication token is valid
   - Ensure the organization and bucket exist

4. **"Timeout when saving symbol to InfluxDB"**:
   - Increase the timeout value with `--timeout` parameter
   - Check InfluxDB server load or network connectivity
   - Data will be saved to fallback files if InfluxDB is unreachable

5. **"Operation doesn't complete"**:
   - Set a maximum runtime with `--timeout 300` (5 minutes)
   - Use a smaller batch of symbols or reduce concurrency
   - Press Ctrl+C for graceful shutdown at any time

### Performance Recommendations

- For a few symbols: use `--batch-size 5 --max-concurrent 2`
- For many symbols: use `--batch-size 20 --max-concurrent 5`
- For very large lists: use `--batch-size 50 --max-concurrent 10 --rate-limit 1.0 --timeout 1800`

### Handling Long-Running Operations

For very large symbol lists (100+ symbols):
1. Use the `--timeout` parameter to automatically terminate after a set time
2. Use Ctrl+C for graceful shutdown at any time (will save already-fetched data)
3. Consider splitting your symbol list into smaller batches
4. Check InfluxDB settings and optimize if needed 
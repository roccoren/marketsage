# Stock Data Fetcher

A tool for downloading historical daily price data for NYSE and NASDAQ listed stocks.

## Features

- Downloads historical price data from Yahoo Finance
- Supports both NYSE and NASDAQ markets
- Built-in rate limiting and caching
- Organized data storage by market/symbol/date
- Batch processing for efficient downloads
- Error handling and recovery

## Usage

1. Ensure you have the required JSON files with stock symbols:
   - `nyse-listed.json` for NYSE stocks
   - `nasdaq-listed.json` for NASDAQ stocks

2. Run the fetcher:

```python
from datetime import datetime
from market_data_fetcher import MarketDataFetcher
import asyncio

async def main():
    fetcher = MarketDataFetcher(cache_dir="data")
    
    # Set your desired start date
    start_date = datetime(2024, 1, 1)
    
    # Fetch data for both markets
    await fetcher.fetch_market_data("nyse", "nyse-listed.json", start_date)
    await fetcher.fetch_market_data("nasdaq", "nasdaq-listed.json", start_date)

if __name__ == "__main__":
    asyncio.run(main())
```

## Data Storage

The downloaded data is stored in the following structure:

```
data/
  ├── nyse/
  │   ├── AAPL/
  │   │   ├── 2024-01-02.json
  │   │   ├── 2024-01-03.json
  │   │   └── ...
  │   └── ...
  └── nasdaq/
      ├── MSFT/
      │   ├── 2024-01-02.json
      │   ├── 2024-01-03.json
      │   └── ...
      └── ...
```

## Configuration

The fetcher uses conservative rate limiting by default:
- 2 requests per second
- Burst size of 5 requests
- 60-second retry delay after hitting limits
- 24-hour cache TTL

You can modify these settings by adjusting the parameters in the `_setup_provider` method.
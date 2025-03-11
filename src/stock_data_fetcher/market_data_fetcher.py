import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import (
    ProviderConfig,
    RateLimitConfig,
    CacheConfig,
    MarketData
)
from market_data.exceptions import (
    MarketDataError,
    RateLimitError,
    CircuitBreakerError,
    ValidationError
)

class MarketDataFetcher:
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        self.provider = self._setup_provider()
        
    def _setup_provider(self) -> YahooFinanceProvider:
        """Initialize the Yahoo Finance provider with optimized configuration."""
        config = ProviderConfig(
            name="yahoo",
            rate_limit=RateLimitConfig(
                requests_per_second=2.0,  # Conservative rate limiting
                burst_size=5,
                retry_after=60
            ),
            cache=CacheConfig(
                enabled=True,
                ttl=86400,  # 24 hours
                directory=".cache"
            )
        )
        return YahooFinanceProvider(config)
    
    def _load_symbols(self, file_path: str) -> List[str]:
        """Load stock symbols from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [item['symbol'] for item in data]
        except FileNotFoundError:
            print(f"Symbol file not found: {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")
            return []
        except KeyError:
            print(f"Unexpected JSON structure in file: {file_path}")
            return []

    async def _save_market_data(self, market: str, symbol: str, date: datetime, data: MarketData):
        """Save market data to the appropriate directory structure."""
        # Create directory structure
        symbol_dir = Path(self.cache_dir) / market / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert MarketData to dictionary format
        data_dict = {
            "open": float(data.open),
            "high": float(data.high),
            "low": float(data.low),
            "close": float(data.close),
            "volume": float(data.volume),
            "dividends": float(data.dividends or 0),
            "stock_splits": float(data.stock_splits or 0)
        }
        
        # Save data to date-specific file
        file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2)

    async def fetch_market_data(self, market: str, symbols_file: str, start_date: datetime):
        """
        Fetch historical data for all symbols in a market from the given start date.
        
        Args:
            market: Market name (e.g., 'nyse', 'nasdaq')
            symbols_file: Path to the JSON file containing symbols
            start_date: Start date for historical data
        """
        symbols = self._load_symbols(symbols_file)
        if not symbols:
            print(f"No symbols found for market: {market}")
            return

        print(f"Fetching data for {len(symbols)} symbols from {market}")
        
        # Process symbols in batches
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                # Fetch batch data using the optimized batch fetching capability
                batch_data = await self.provider.get_history_batch(
                    batch,
                    start_date,
                    datetime.now()
                )
                
                # Save data for each symbol
                for symbol, result in batch_data.items():
                    if result and result.data:
                        for date, data_point in result.data.items():
                            await self._save_market_data(
                                market,
                                symbol,
                                date,
                                data_point
                            )
                        print(f"Saved data for {symbol}")
                    else:
                        print(f"No data available for {symbol}")
                        
            except RateLimitError:
                print(f"Rate limit exceeded for batch {i//batch_size + 1}. Waiting...")
                await asyncio.sleep(60)  # Wait for rate limit to reset
                continue
            except CircuitBreakerError:
                print(f"Service temporarily unavailable for batch {i//batch_size + 1}")
                await asyncio.sleep(300)  # Wait longer for service recovery
                continue
            except ValidationError as e:
                print(f"Validation error in batch {i//batch_size + 1}: {str(e)}")
                continue
            except MarketDataError as e:
                print(f"Error fetching batch {i//batch_size + 1}: {str(e)}")
                continue
                
            # Small delay between batches for rate limiting
            await asyncio.sleep(1)

async def main():
    fetcher = MarketDataFetcher()
    start_date = datetime(2024, 1, 1)  # Example start date
    
    # Fetch NYSE data
    await fetcher.fetch_market_data(
        "nyse",
        "nyse-listed.json",
        start_date
    )
    
    # Fetch NASDAQ data
    await fetcher.fetch_market_data(
        "nasdaq",
        "nasdaq-listed.json",
        start_date
    )

if __name__ == "__main__":
    asyncio.run(main())
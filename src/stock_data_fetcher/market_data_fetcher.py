import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        logger.debug("Setting up Yahoo Finance provider")
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
        provider = YahooFinanceProvider(config)
        logger.debug("Yahoo Finance provider initialized successfully")
        return provider
    
    def _load_symbols(self, file_path: str) -> List[str]:
        """Load stock symbols from a JSON file."""
        try:
            logger.debug(f"Attempting to load symbols from: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                symbols = [item['symbol'] for item in data]
                logger.debug(f"Successfully loaded symbols: {symbols}")
                return symbols
        except FileNotFoundError:
            logger.error(f"Symbol file not found: {file_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {file_path}")
            return []
        except KeyError:
            logger.error(f"Unexpected JSON structure in file: {file_path}")
            return []

    async def _save_market_data(self, market: str, symbol: str, date: datetime, data: MarketData):
        """Save market data to the appropriate directory structure."""
        try:
            # Create directory structure
            symbol_dir = Path(self.cache_dir) / market / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {symbol_dir}")
            
            # Convert MarketData to dictionary format
            data_dict = {
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "close": float(data.close),
                "volume": float(data.volume),
                "dividends": float(getattr(data, "dividends", 0) or 0),
                "stock_splits": float(getattr(data, "stock_splits", 0) or 0)
            }
            logger.debug(f"Data for {symbol} on {date}: {data_dict}")
            
            # Save data to date-specific file
            file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
            logger.debug(f"Attempting to save to file (absolute): {file_path.resolve()}")
            logger.debug(f"Data to save: {data_dict}")
            try:
                with open(file_path, 'w') as f:
                    json.dump(data_dict, f, indent=2)
                logger.debug(f"Successfully saved data to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save data to file: {e}")
                logger.error(f"File path: {file_path}")
                logger.error(f"Directory exists: {symbol_dir.exists()}")
                logger.error(f"Directory is writable: {os.access(str(symbol_dir), os.W_OK)}")
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    async def fetch_market_data(self, market: str, symbols_file: str, start_date: datetime):
        """
        Fetch historical data for all symbols in a market from the given start date.
        
        Args:
            market: Market name (e.g., 'nyse', 'nasdaq')
            symbols_file: Path to the JSON file containing symbols
            start_date: Start date for historical data
        """
        logger.info(f"Starting market data fetch for {market}")
        logger.debug(f"Using symbols file: {symbols_file}")
        logger.debug(f"Start date: {start_date}")

        symbols = self._load_symbols(symbols_file)
        if not symbols:
            logger.warning(f"No symbols found for market: {market}")
            return

        logger.info(f"Fetching data for {len(symbols)} symbols from {market}")
        
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
                    logger.debug(f"Processing result for {symbol}: {type(result)}")
                    if result and result.data:  # result is HistoricalData object
                        logger.debug(f"Result data: {result}")
                        for data_point in result.data:
                            await self._save_market_data(
                                market,
                                symbol,
                                data_point.date,
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
            except Exception as e:
                import traceback
                logger.error(f"Error fetching batch {i//batch_size + 1}:")
                logger.error(traceback.format_exc())
                continue
                
            # Small delay between batches for rate limiting
            await asyncio.sleep(1)

async def main():
    fetcher = MarketDataFetcher(cache_dir="test_data")
    start_date = datetime(2024, 1, 1)  # Example start date
    
    # Fetch test data for sample symbols
    await fetcher.fetch_market_data(
        "nasdaq",
        "src/stock_data_fetcher/test-symbols.json",
        start_date
    )

if __name__ == "__main__":
    asyncio.run(main())
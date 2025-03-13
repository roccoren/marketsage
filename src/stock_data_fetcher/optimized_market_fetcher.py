import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Dict, Optional
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import time
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMarketFetcher:
    """
    Optimized market data fetcher with:
    - Batch symbol processing using yfinance.download
    - Async/parallel processing with controlled concurrency
    - Intelligent rate limiting and retry logic
    - Efficient data caching
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "/datadrive/workspaces/stockscan-market-fetch/config/market",
        max_concurrent: int = 3,
        batch_size: int = 10,
        rate_limit_delay: float = 2.0
    ):
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Concurrency controls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # Cache settings
        self.cache_dir = self.data_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(self.cache_dir))
        
        # Initialize markets data
        self.markets: Dict[str, Set[str]] = {}
        self._load_markets()

    def _load_markets(self) -> None:
        """Load market symbol listings from config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
            
        for file_path in self.config_dir.glob("*-listed.json"):
            market_name = file_path.stem.replace("-listed", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    symbols = {
                        item.get('Symbol', item.get('symbol'))
                        for item in data
                        if 'Symbol' in item or 'symbol' in item
                    }
                    if symbols:
                        self.markets[market_name] = symbols
                        logger.info(f"Loaded {len(symbols)} symbols for {market_name}")
            except Exception as e:
                logger.error(f"Error loading market {market_name}: {str(e)}")

    async def _download_batch(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Download data for a batch of symbols using yfinance."""
        if not symbols:
            return None
            
        async with self.semaphore:
            try:
                loop = asyncio.get_running_loop()
                download_func = partial(
                    yf.download,
                    tickers=" ".join(symbols),
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=False,
                    threads=False  # Disable internal threading as we're managing concurrency
                )
                
                df = await loop.run_in_executor(self.thread_pool, download_func)
                
                if isinstance(df.columns, pd.MultiIndex):
                    return df
                elif not df.empty:
                    # Handle single symbol case
                    df.columns = pd.MultiIndex.from_product([[symbols[0]], df.columns])
                    return df
                    
                return None
                
            except Exception as e:
                logger.error(f"Error downloading batch {symbols}: {str(e)}")
                return None
            finally:
                # Rate limiting delay
                await asyncio.sleep(self.rate_limit_delay)

    async def _save_symbol_data(self, market: str, symbol: str, data: pd.DataFrame):
        """Save market data for a symbol with async file operations."""
        symbol_dir = self.data_dir / market / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert MultiIndex DataFrame to dict format
        symbol_data = data[symbol]
        
        dates = symbol_data.index
        for date in dates:
            row = symbol_data.loc[date]
            
            # Skip if any essential data is missing
            if row.isna().any():
                continue
                
            data_dict = {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "adjusted_close": float(row["Close"])  # Using adjusted data
            }
            
            file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data_dict, indent=2))

    async def fetch_market_data(
        self,
        market: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Fetch market data with optimized batch processing.
        
        Args:
            market: Market name (e.g., "nasdaq", "nyse")
            start_date: Start date for data fetching
            end_date: Optional end date (defaults to current date)
            symbols: Optional list of specific symbols to fetch (defaults to all market symbols)
        """
        if market not in self.markets:
            raise ValueError(f"Invalid market: {market}")
            
        end_date = end_date or datetime.now()
        target_symbols = symbols or list(self.markets[market])
        
        logger.info(f"Fetching data for {len(target_symbols)} symbols from {market}")
        
        # Process symbols in batches
        batches = [
            target_symbols[i:i + self.batch_size]
            for i in range(0, len(target_symbols), self.batch_size)
        ]
        
        total_batches = len(batches)
        for batch_idx, batch in enumerate(batches, 1):
            try:
                logger.info(f"Processing batch {batch_idx}/{total_batches} for {market}")
                
                df = await self._download_batch(batch, start_date, end_date)
                if df is not None and not df.empty:
                    # Process and save data for each symbol in the batch
                    save_tasks = []
                    for symbol in batch:
                        if symbol in df.columns.levels[0]:
                            save_tasks.append(
                                self._save_symbol_data(market, symbol, df)
                            )
                    
                    if save_tasks:
                        await asyncio.gather(*save_tasks)
                        
                    logger.info(f"Completed batch {batch_idx}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    def get_available_markets(self) -> List[str]:
        """Get list of available markets."""
        return list(self.markets.keys())

    def get_market_symbols(self, market: str) -> Set[str]:
        """Get set of symbols for a market."""
        return self.markets.get(market, set())

async def main():
    # Example usage
    fetcher = OptimizedMarketFetcher(
        max_concurrent=3,        # Maximum concurrent downloads
        batch_size=10,          # Symbols per batch
        rate_limit_delay=2.0    # Seconds between requests
    )
    
    # Fetch last 7 days of data
    start_date = datetime.now() - timedelta(days=7)
    
    # Example 1: Fetch all NASDAQ symbols
    await fetcher.fetch_market_data("nasdaq", start_date)
    
    # Example 2: Fetch specific symbols
    specific_symbols = ["AAPL", "MSFT", "GOOGL"]
    await fetcher.fetch_market_data("nasdaq", start_date, symbols=specific_symbols)

if __name__ == "__main__":
    asyncio.run(main())
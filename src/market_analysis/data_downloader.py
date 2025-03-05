"""
Stock Data Downloader Module

Downloads stock data from various sources and saves it in JSON format.
Currently supports Yahoo Finance as the data source.
"""
import os
import json
import datetime
import time
import signal
import logging
import threading
import asyncio
import aiohttp
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, TypeVar, Any
import yfinance as yf
from pandas import DataFrame, Series
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import partial, lru_cache
import aiofiles
import msgpack
from contextlib import asynccontextmanager

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
root_handler = root_logger.hasHandlers() and root_logger.handlers[0]
if root_handler:
    root_logger.removeHandler(root_handler)
root_logger.addHandler(console_handler)

# Add file handler to root logger
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/stock_downloader.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

# Custom exceptions
class StockDataError(Exception):
    """Base exception for stock data operations."""
    pass

class RateLimitError(StockDataError):
    """Raised when API rate limit is exceeded."""
    pass

class DataDownloadError(StockDataError):
    """Raised when data download fails."""
    pass

@dataclass
class StockData:
    """Standardized container for stock market data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_yfinance(cls, date: str, data: Series) -> 'StockData':
        """Create StockData instance from yfinance data."""
        return cls(
            date=date,
            open=float(data["Open"]),
            high=float(data["High"]),
            low=float(data["Low"]),
            close=float(data["Close"]),
            volume=float(data["Volume"])
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }

def print_progress(current, total, prefix='Progress:', suffix='Complete'):
    """Print progress bar."""
    length = 50
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    percent = current / total * 100
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='\r')
    if current == total:
        print()

def print_summary(market: str, results: dict, start_time: float):
    """Print a formatted summary of market processing results."""
    elapsed = time.time() - start_time
    successful = len([v for v in results.values() if v])
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print("\n" + "="*80)
    print(f"Market Summary: {market}")
    print("="*80)
    print(f"Total symbols processed: {total}")
    print(f"Successfully updated:    {successful}")
    print(f"Failed:                 {total - successful}")
    print(f"Success rate:           {success_rate:.1f}%")
    print(f"Time elapsed:           {elapsed:.1f} seconds")
    print(f"Average time/symbol:    {(elapsed/total):.2f} seconds")
    print("="*80 + "\n")

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_second=5):
        self.rate = calls_per_second
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to maintain rate limit."""
        async with self.lock:
            now = time.time()
            if self.last_call:
                time_since_last = now - self.last_call
                if time_since_last < self.minimum_interval:
                    await asyncio.sleep(self.minimum_interval - time_since_last)
            self.last_call = time.time()

class StockDataDownloader:
    """Downloads stock data from various markets and saves it in optimized format."""
    
    MARKET_CODES = {
        "NYSE": "",
        "NASDAQ": "",
        "ASX": ".AX",
        "HKEX": ".HK"
    }
    
    def __init__(self, base_data_dir: str = None, max_workers: int = None):
        """Initialize the downloader with base directory for data storage."""
        self.base_data_dir = Path(base_data_dir) if base_data_dir else Path(__file__).parent.parent.parent / "data"
        logger.info(f"Initializing StockDataDownloader with base_dir: {self.base_data_dir}, max_workers: {max_workers}")
        self.cache_dir = self.base_data_dir / "yfinance_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yfinance
        yf.set_tz_cache_location(str(self.cache_dir))
        
        # Initialize parameters
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.rate_limiter = RateLimiter(calls_per_second=5)
        
        # Set up connection pool
        self.session = None
        self.connector = None

    @asynccontextmanager
    async def get_session(self):
        """Get or create an aiohttp session with connection pooling."""
        logger.debug("Session status - "
                    f"Exists: {self.session is not None}, "
                    f"Is closed: {self.session.closed if self.session else True}")
        if self.session is None or self.session.closed:
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=self.max_workers,
                    ttl_dns_cache=300,
                    use_dns_cache=True
                )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        try:
            yield self.session
        finally:
            pass  # Keep session open for reuse

    async def _download_stock_data_async(
        self,
        full_ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        market: str,
        save_dir: Optional[str] = None,
        save_daily: bool = True
    ) -> Dict[str, Any]:
        """Download stock data with optimized async operations."""
        start_time = time.time()
        logger.info(f"Attempting to download data for {full_ticker} at {datetime.datetime.now()}")
        await self.rate_limiter.acquire()

        retries = 3
        retry_delay = 1

        try:
            for attempt in range(retries):
                try:
                    # Use ThreadPoolExecutor for CPU-bound yfinance operations
                    loop = asyncio.get_running_loop()
                    logger.debug(f"Event loop status for {full_ticker} - Is running: {loop.is_running()}")
                    logger.info(f"Creating yfinance.Ticker for {full_ticker}")
                    stock = yf.Ticker(full_ticker)
                    logger.info(f"Fetching history for {full_ticker} from {start_date} to {end_date}")

                    df = await asyncio.wait_for(
                        loop.run_in_executor(None, partial(stock.history, start=start_date, end=end_date)),
                        timeout=30.0  # 30 second timeout for each stock
                    )

                    duration = time.time() - start_time
                    logger.info(f"Retrieved {len(df)} data points for {full_ticker} in {duration:.2f} seconds")
                    
                    if df.empty:
                        if attempt < retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"No data retrieved for {full_ticker}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        logger.warning(f"No data retrieved for {full_ticker} after {duration:.2f} seconds")
                        raise DataDownloadError(f"No data retrieved for {full_ticker}")

                    # Prepare directories
                    market_dir = Path(save_dir or self.base_data_dir) / market
                    market_dir.mkdir(parents=True, exist_ok=True)
                    ticker_dir = market_dir / full_ticker.split('.')[0]
                    ticker_dir.mkdir(parents=True, exist_ok=True)

                    # Process data
                    stock_data = []
                    for date, day_data in df.iterrows():
                        data = StockData.from_yfinance(date.strftime("%Y-%m-%d"), day_data)
                        stock_data.append(data.to_dict())

                    # Batch file operations
                    tasks = []
                    
                    if save_daily:
                        # Group by month for more efficient storage
                        monthly_data = {}
                        for data in stock_data:
                            month = data['date'][:7]  # YYYY-MM
                            monthly_data.setdefault(month, []).append(data)
                        
                        for month, data in monthly_data.items():
                            month_file = ticker_dir / f"{month}.msgpack"
                            tasks.append(self._save_data_async(month_file, data))

                    # Save complete history using msgpack for better performance
                    history_file = ticker_dir / "history.msgpack"
                    tasks.append(self._save_data_async(history_file, stock_data))
                    
                    # Execute all file operations concurrently
                    await asyncio.gather(*tasks)
                    
                    return {"history": stock_data}
                except asyncio.TimeoutError:
                    if attempt < retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Timeout for {full_ticker}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error(f"Timeout while fetching data for {full_ticker}")
                    raise DataDownloadError(f"Timeout while fetching data for {full_ticker}")
            return None
        except Exception as e:
            if isinstance(e, (aiohttp.ClientError, yf.exceptions.YFinanceError)):
                raise RateLimitError(f"Rate limit exceeded for {full_ticker}")
            raise DataDownloadError(f"Failed to download {full_ticker} data: {str(e)}")
        return None

    @staticmethod
    async def _save_data_async(file_path: Path, data: List[dict]):
        """Save data asynchronously using msgpack for better performance."""
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(msgpack.packb(data, use_bin_type=True))

    @staticmethod
    async def _load_data_async(file_path: Path) -> List[dict]:
        """Load data asynchronously using msgpack."""
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            return msgpack.unpackb(content, raw=False)

    async def download_multiple_stocks_async(
        self,
        tickers: List[str],
        market: str,
        start_date: Union[str, datetime.date],
        end_date: Optional[Union[str, datetime.date]] = None,
        save_dir: Optional[str] = None,
        save_daily: bool = True,
        batch_size: int = 20
    ) -> Dict[str, Dict]:
        """Download multiple stocks with optimized batch processing."""
        batch_start_time = time.time()
        logger.info(f"Starting async download for {len(tickers)} tickers")
        logger.info(f"Market: {market}, Start date: {start_date}, End date: {end_date}")
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

        process_start_time = time.time()
        if end_date is None:
            end_date = datetime.date.today()
        elif isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        results = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        total_processed = 0

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tickers))
            batch_tickers = tickers[start_idx:end_idx]
            current_time = datetime.datetime.now()
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch_tickers)} tickers at {current_time}")
            
            tasks = [
                self._download_stock_data_async(
                    f"{ticker}{self.MARKET_CODES[market]}",
                    start_date,
                    end_date,
                    market.upper(),
                    save_dir,
                    save_daily
                ) for ticker in batch_tickers
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_duration = time.time() - batch_start_time
            total_processed += len([r for r in batch_results if not isinstance(r, Exception)])
            print_progress(total_processed, len(tickers), prefix=f'{market} Progress:')
            logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f} seconds")
            
            for ticker, result in zip(batch_tickers, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error downloading {ticker}: {str(result)}")
                    if isinstance(result, yf.exceptions.YFinanceError):
                        logger.error(f"YFinance error details for {ticker}: {str(result)}")
                else:
                    results[ticker] = result
            
            logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
            
        print_summary(market, results, process_start_time)
            
        return results

    def download_multiple_stocks(
        self,
        tickers: List[str],
        market: str,
        start_date: Union[str, datetime.date],
        end_date: Optional[Union[str, datetime.date]] = None,
        save_dir: Optional[str] = None,
        save_daily: bool = True,
        batch_size: int = 20
    ) -> Dict[str, Dict]:
        logger.info(f"Processing download request for {len(tickers)} tickers")
        logger.info(f"Available market codes: {self.MARKET_CODES}")
        """Synchronous interface for multiple stock downloads."""
        try:
            logger.info("Initializing event loop")
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.info("No event loop found, creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.download_multiple_stocks_async(
                    tickers=tickers,
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    save_dir=save_dir,
                    save_daily=save_daily,
                    batch_size=batch_size
                )
            )
        except Exception as e:
            logger.error(f"Error in download_multiple_stocks: {str(e)}")
            raise

if __name__ == "__main__":
    downloader = StockDataDownloader()
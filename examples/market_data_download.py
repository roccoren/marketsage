"""
Market Data Downloader Script
Downloads stock data for symbols listed in NYSE and NASDAQ JSON files using parallel processing.
Uses the StockDataDownloader class to efficiently download and store historical data.
"""
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time
from typing import Dict, List, Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from src.market_analysis.data_downloader import StockDataDownloader

# Enhanced logging configuration
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
root_handler = root_logger.hasHandlers() and root_logger.handlers[0]
if root_handler:
    root_logger.removeHandler(root_handler)
root_logger.addHandler(console_handler)

# Add file handler
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/market_download.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

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

    logger.info(f"\nMarket {market} completed:")
    logger.info(f"Success rate: {success_rate:.1f}% ({successful}/{total})")
    logger.info(f"Time elapsed: {elapsed:.1f}s")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/market_download.log')
    ]
)

class MarketDataManager:
    """Manages the downloading of market data with progress tracking and error handling"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data'
        self.downloader = StockDataDownloader(str(self.data_dir))
        
    def _get_data_date_range(self, market: str, symbol: str) -> tuple[datetime.date, datetime.date]:
        """Get the earliest and latest dates available for a symbol."""
        symbol_dir = self.data_dir / market / symbol
        if not symbol_dir.exists():
            return None, None
            
        # Look for daily JSON files
        dates = []
        for file in symbol_dir.glob("*.json"):
            if file.stem == "history":
                continue
            try:
                dates.append(datetime.datetime.strptime(file.stem, "%Y-%m-%d").date())
            except ValueError:
                continue
                
        if not dates:
            return None, None
            
        return min(dates), max(dates)
        
    def update_stock_data(
        self,
        market: str,
        symbols: List[str] = None,
        batch_size: int = 10
    ) -> Dict:
        """
        Incrementally update market data from the last available date until today.
        
        Args:
            market: Market name (NYSE, NASDAQ)
            symbols: Optional list of specific symbols to update. If None, updates all symbols.
            batch_size: Number of stocks to download in each batch
            
        Returns:
            Dictionary of results with updates performed
        """
        if symbols is None:
            symbols = self.load_symbols(market)
            
        if not symbols:
            logger.warning(f"No symbols provided for {market}")
            return {}
            
        results = {}
        today = datetime.date.today()
        
        for symbol in symbols:
            _, last_date = self._get_data_date_range(market, symbol)
            if last_date is None:
                # No existing data, download full history
                start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
                logger.info(f"No existing data for {symbol}, downloading last 30 days")
            elif last_date >= today:
                logger.info(f"Data for {symbol} is already up to date")
                continue
            else:
                # Start from the day after the last available date
                start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"Updating {symbol} from {start_date} to today")
                
            try:
                symbol_data = self.download_market_data(market, [symbol], start_date, batch_size=batch_size)
                if symbol_data and symbol in symbol_data:
                    logger.info(f"Successfully updated {symbol}")
                    print_progress(total_processed, len(symbols), prefix=f'{market} Progress:')
                    results[symbol] = symbol_data
            except Exception as e:
                logger.error(f"Error updating {symbol}: {str(e)}")
                
        return results
        
    def fetch_historical_gap(
        self,
        market: str,
        start_date: str,
        symbols: List[str] = None,
        batch_size: int = 10
    ) -> Dict:
        """
        Fill historical data gaps between requested start date and earliest available data.
        
        Args:
            market: Market name (NYSE, NASDAQ)
            start_date: Requested start date (YYYY-MM-DD)
            symbols: Optional list of specific symbols. If None, checks all symbols.
            batch_size: Number of stocks to download in each batch
            
        Returns:
            Dictionary of results with gaps filled
        """
        if symbols is None:
            symbols = self.load_symbols(market)
            
        if not symbols:
            logger.warning(f"No symbols provided for {market}")
            return {}
            
        results = {}
        requested_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        
        for symbol in symbols:
            earliest_date, _ = self._get_data_date_range(market, symbol)
            if earliest_date is None:
                # No existing data, download from requested start date
                logger.info(f"No existing data for {symbol}, downloading from {start_date}")
            elif earliest_date <= requested_start:
                logger.info(f"No historical gap for {symbol}, earliest data: {earliest_date}")
                continue
            else:
                logger.info(f"Filling historical gap for {symbol} from {start_date} to {earliest_date}")
                
            try:
                symbol_data = self.download_market_data(market, [symbol], start_date, batch_size=batch_size)
                if symbol_data and symbol in symbol_data:
                    logger.info(f"Successfully filled historical gap for {symbol}")
                    print_progress(total_processed, len(symbols), prefix=f'{market} Historical:')
                    results[symbol] = symbol_data
            except Exception as e:
                logger.error(f"Error filling historical gap for {symbol}: {str(e)}")
                
        return results

    def load_symbols(self, market: str) -> List[str]:
        """Load stock symbols from market listing file with error handling"""
        file_path = self.data_dir / f"{market.lower()}-listed.json"
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                symbols = [item["Symbol"] for item in data if "Symbol" in item]
                logger.info(f"Loaded {len(symbols)} symbols from {market}")
                return symbols
        except FileNotFoundError:
            logger.error(f"Market listing file not found: {file_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading symbols from {file_path}: {str(e)}")
            return []

    def download_market_data(
        self,
        market: str,
        symbols: List[str],
        start_date: str = None,
        batch_size: int = 10,
        save_daily: bool = True
    ) -> Dict:
        """Download data for a list of symbols with progress tracking"""
        if not symbols:
            logger.warning(f"No symbols provided for {market}")
            return {}
            
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting download for {len(symbols)} {market} symbols from {start_date}")
        logger.info(f"Configuration: batch_size={batch_size}")
        
        start_time = time.time()
        total_processed = 0
        try:
            logger.debug(f"Calling StockDataDownloader.download_multiple_stocks with args: market={market}, start_date={start_date}, batch_size={batch_size}, save_daily={save_daily}")
            results = self.downloader.download_multiple_stocks(
                symbols,
                market,
                start_date,
                batch_size=batch_size,
                save_daily=save_daily
            )
            total_processed = len([r for r in results.values() if r is not None])
            print_progress(total_processed, len(symbols), prefix=f'{market} Progress:')
            
            # Print summary
            print_summary(market, results, start_time)
            
            logger.info(f"Download complete for {market}")
            return results
            
        except Exception as e:
            logger.error(f"Error downloading {market} data: {str(e)}")
            return {}

    def update_to_current(self,
                         market: str,
                         symbols: Optional[List[str]] = None,
                         batch_size: int = 10) -> Dict[str, Dict]:
        """
        Update market data for specified symbols from their last available date to today.

        This function checks each symbol's most recent data and downloads any missing data
        up to today's date. For example, if a symbol's latest data is from March 1st and 
        today is March 4th, it will download data for March 1st through March 4th.

        Args:
            market: Market name (NYSE, NASDAQ)
            symbols: Optional list of specific symbols to update. If None, updates all symbols.
            batch_size: Number of stocks to download in each batch (default: 10)

        Returns:
            Dictionary mapping symbols to their updated data

        Example:
            >>> manager = MarketDataManager()
            >>> updated_data = manager.update_to_current("NYSE", ["AAPL", "MSFT"])
        """
        try:
            if symbols is None:
                symbols = self.load_symbols(market)
                if not symbols:
                    logger.warning(f"No symbols loaded for {market}")
                    return {}
            
            logger.info(f"Starting update to current date for {len(symbols)} {market} symbols")
            start_time = time.time()
            
            results = self.downloader.update_to_current(market, symbols, batch_size)
            
            elapsed = time.time() - start_time
            print_summary(market, results, start_time)

            logger.info(f"Time taken: {elapsed:.2f} seconds")
            
            return results
        except Exception as e:
            logger.error(f"Error updating {market} data to current: {str(e)}")
            return {}

    def backfill_historical_data(self,
                               market: str,
                               start_date: str,
                               symbols: Optional[List[str]] = None,
                               batch_size: int = 10) -> Dict[str, Dict]:
        """
        Fill historical data gaps between start_date and earliest existing data.

        For example, if start_date is '2020-11-01' and the earliest data for a symbol
        is from '2021-01-01', this will fetch data for 2020-11-01 to 2021-01-01.

        Args:
            market: Market name (NYSE, NASDAQ)
            start_date: Start date for historical data in YYYY-MM-DD format
            symbols: Optional list of specific symbols to backfill. If None, checks all symbols.
            batch_size: Number of stocks to download in each batch (default: 10)

        Returns:
            Dictionary mapping symbols to their backfilled data

        Example:
            >>> manager = MarketDataManager()
            >>> historical_data = manager.backfill_historical_data(
            ...     "NYSE", "2020-11-01", ["AAPL", "MSFT"]
            ... )
        """
        try:
            if symbols is None:
                symbols = self.load_symbols(market)
                if not symbols:
                    logger.warning(f"No symbols loaded for {market}")
                    return {}
            
            logger.info(f"Starting historical backfill from {start_date} for {len(symbols)} {market} symbols")
            start_time = time.time()
            
            results = self.downloader.backfill_historical_data(market, start_date, symbols, batch_size)
            
            print_summary(market, results, start_time)
            return results            
        except Exception as e:
            logger.error(f"Error backfilling {market} historical data: {str(e)}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Download stock data from NYSE and NASDAQ listings')
    
    parser.add_argument('--start', '-s', type=str, default=None,
                       help='Start date in YYYY-MM-DD format (default: 30 days ago)')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                       help='Number of stocks to download in each batch (default: 10)')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker threads for parallel downloading')
    parser.add_argument('--no-daily', '-nd', action='store_false', dest='save_daily',
                       help='Do not save individual daily files')
    parser.add_argument('--market', '-m', choices=['NYSE', 'NASDAQ', 'ALL'],
                       default='ALL', help='Market to download (default: ALL)')
    parser.add_argument('--update-current', '-u', action='store_true',
                       help='Update data to current date for existing symbols')
    parser.add_argument('--backfill', '-bf', type=str, metavar='START_DATE',
                       help='Backfill historical data from START_DATE (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to process (optional)')
    
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = MarketDataManager()
    
    # Track overall statistics
    total_results = {}
    overall_start = time.time()
    
    markets_to_process = ['NYSE', 'NASDAQ'] if args.market == 'ALL' else [args.market]
    
    try:
        for market in markets_to_process:
            if args.update_current:
                logger.info(f"\n=== Updating {market} data to current date ===")
                results = manager.update_to_current(
                    market,
                    args.symbols,
                    args.batch_size
                )
                if results:
                    total_results[market] = results
            
            elif args.backfill:
                logger.info(f"\n=== Backfilling {market} historical data from {args.backfill} ===")
                results = manager.backfill_historical_data(
                    market,
                    args.backfill,
                    args.symbols,
                    args.batch_size
                )
                if results:
                    total_results[market] = results
            
            else:
                # Regular data download
                symbols = args.symbols or manager.load_symbols(market)
                if not symbols:
                    continue
                
                results = manager.download_market_data(
                    market,
                    symbols,
                    args.start,
                    args.batch_size,
                    args.save_daily
                )
                
                if results:
                    total_results[market] = results
    
    except Exception as e:
        logger.error(f"Error processing markets: {str(e)}")
        sys.exit(1)
    
    # Print final summary
    total_elapsed = time.time() - overall_start
    total_stocks = sum(len(results) for results in total_results.values())
    
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total stocks downloaded: {total_stocks}")
    logger.info(f"Total time taken: {total_elapsed:.2f} seconds")
    
    for market, results in total_results.items():
        success_rate = len(results) / len(manager.load_symbols(market)) * 100
        logger.info(f"{market}: {len(results)} stocks ({success_rate:.1f}% success rate)")

if __name__ == "__main__":
    main()
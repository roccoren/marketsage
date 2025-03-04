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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_download.log')
    ]
)
logger = logging.getLogger(__name__)

class MarketDataManager:
    """Manages the downloading of market data with progress tracking and error handling"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data'
        self.downloader = StockDataDownloader(str(self.data_dir))
        
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
        parallel: bool = True,
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
        logger.info(f"Configuration: parallel={parallel}, batch_size={batch_size}")
        
        start_time = time.time()
        try:
            results = self.downloader.download_multiple_stocks(
                symbols,
                market,
                start_date,
                parallel=parallel,
                batch_size=batch_size,
                save_daily=save_daily
            )
            
            elapsed = time.time() - start_time
            success_rate = len(results) / len(symbols) * 100
            
            logger.info(f"Download complete for {market}")
            logger.info(f"Successfully downloaded {len(results)}/{len(symbols)} symbols ({success_rate:.1f}%)")
            logger.info(f"Time taken: {elapsed:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error downloading {market} data: {str(e)}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Download stock data from NYSE and NASDAQ listings')
    
    parser.add_argument('--start', '-s', type=str, default=None,
                       help='Start date in YYYY-MM-DD format (default: 30 days ago)')
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                       help='Use parallel downloading (default: True)')
    parser.add_argument('--no-parallel', '-np', action='store_false', dest='parallel',
                       help='Disable parallel downloading')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                       help='Number of stocks to download in each batch (default: 10)')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker threads for parallel downloading')
    parser.add_argument('--no-daily', '-nd', action='store_false', dest='save_daily',
                       help='Do not save individual daily files')
    parser.add_argument('--market', '-m', choices=['NYSE', 'NASDAQ', 'ALL'],
                       default='ALL', help='Market to download (default: ALL)')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = MarketDataManager()
    
    # Track overall statistics
    total_results = {}
    overall_start = time.time()
    
    markets_to_process = ['NYSE', 'NASDAQ'] if args.market == 'ALL' else [args.market]
    
    for market in markets_to_process:
        # Load symbols
        symbols = manager.load_symbols(market)
        if not symbols:
            continue
            
        # Download data
        results = manager.download_market_data(
            market,
            symbols,
            args.start,
            args.parallel,
            args.batch_size,
            args.save_daily
        )
        
        if results:
            total_results[market] = results
    
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
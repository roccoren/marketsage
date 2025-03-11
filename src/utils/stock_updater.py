#!/usr/bin/env python
"""
Stock Data Updater - Reliable Yahoo Finance data downloader
This script downloads stock data from Yahoo Finance with improved error handling
and rate limiting to avoid connection issues.
"""

import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import traceback
import signal

# Make sure our configuration is loaded first
from . import yfinance_config as config
# Import our improved utilities
from . import yfinance_utils as yfu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.DEFAULT_CONFIG['log_format'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.DEFAULT_CONFIG['log_directory'], 'stock_updater.log'), 
                           mode='w')
    ]
)

logger = logging.getLogger(__name__)

# Global flags for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle interrupt signals by setting global shutdown flag."""
    global shutdown_requested
    logger.info("Shutdown requested. Finishing current batch before exiting...")
    shutdown_requested = True

def load_listing_file(file_path: str) -> List[str]:
    """Load and parse a stock listing file."""
    logger.info(f"Loading symbols from {file_path}")
    
    symbols = yfu.extract_symbols_from_listing(file_path)
    
    if symbols:
        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols
    else:
        logger.error(f"No valid symbols found in {file_path}")
        return []

def filter_problematic_symbols(symbols: List[str]) -> List[str]:
    """Remove known problematic symbols."""
    problematic = {
        'A', 'AA', 'AAM', 'AAM.W', 'AACT.W', 'AACT.U',
        'BMN', 'BMN.US',  # Identified from errors
        'VEEV',           # Identified from errors
        'CNI',            # Identified from errors
        'NQP',           # Identified from errors
        'T', 'F'         # Common problematic symbols
    }
    
    filtered = [s for s in symbols if s not in problematic]
    removed = len(symbols) - len(filtered)
    
    if removed > 0:
        logger.info(f"Filtered out {removed} known problematic symbols")
        
    return filtered

def get_exchange_file(exchange: str) -> Optional[str]:
    """Get the file path for an exchange listing."""
    exchange_files = {
        'NYSE': 'config/nyse-listed.json',
        'NASDAQ': 'config/nasdaq-listed.json',
        'ASX': 'config/asx-listed.json',
        'HKEX': 'config/hkex-listed.json'
    }
    
    if exchange not in exchange_files:
        logger.error(f"Unknown exchange: {exchange}")
        return None
        
    file_path = exchange_files[exchange]
    
    if not os.path.exists(file_path):
        logger.error(f"Exchange listing file not found: {file_path}")
        return None
        
    return file_path

def update_exchange(
    exchange: str, 
    start_date: str,
    end_date: Optional[str] = None,
    batch_size: int = 3,
    workers: int = 1,
    skip_problematic: bool = True,
    max_symbols: Optional[int] = None
) -> bool:
    """Update stock data for an entire exchange."""
    exchange_file = get_exchange_file(exchange)
    
    if not exchange_file:
        return False
        
    # Load symbols
    symbols = load_listing_file(exchange_file)
    
    if not symbols:
        return False
        
    if skip_problematic:
        symbols = filter_problematic_symbols(symbols)
        
    # Limit symbols if requested (for testing)
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]
        logger.info(f"Limited to first {max_symbols} symbols for testing")
    
    # Ensure data directory exists
    data_dir = config.DEFAULT_CONFIG['data_directory']
    os.makedirs(data_dir, exist_ok=True)
    
    # Display configuration
    logger.info(f"Updating {exchange} with {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date or 'today'}")
    logger.info(f"Batch size: {batch_size}, Workers: {workers}")
    logger.info(f"Data directory: {data_dir}")
    
    # Process the exchange
    try:
        start_time = time.time()
        success_count, total = yfu.update_exchange_data(
            exchange=exchange,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            workers=workers,
            batch_size=batch_size
        )
        
        elapsed = time.time() - start_time
        success_rate = (success_count / total) * 100 if total else 0
        
        # Print summary
        logger.info(f"\nUpdate Summary for {exchange}:")
        logger.info(f"Total symbols: {total}")
        logger.info(f"Successfully updated: {success_count} ({success_rate:.1f}%)")
        logger.info(f"Failed: {total - success_count}")
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Average time per symbol: {elapsed / total:.2f} seconds")
        
        # Create filtered listing file with successful symbols
        if success_count > 0:
            # Get successful symbols (ones that have data files)
            exchange_dir = os.path.join(data_dir, exchange)
            successful_symbols = []
            
            for symbol in symbols:
                file_path = os.path.join(exchange_dir, f"{symbol}.json")
                if os.path.exists(file_path):
                    successful_symbols.append(symbol)
            
            # Save filtered listing
            filtered_file = os.path.join(data_dir, f"{exchange.lower()}_filtered_listings.json")
            with open(filtered_file, 'w', encoding='utf-8') as f:
                json.dump(successful_symbols, f, indent=2)
                
            logger.info(f"Created filtered listing file: {filtered_file}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing {exchange}: {str(e)}")
        traceback.print_exc()
        return False

def update_single_symbol(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    exchange: str = 'UNKNOWN'
) -> bool:
    """Update a single stock symbol."""
    logger.info(f"Processing single symbol: {symbol}")
    
    try:
        # Clean symbol
        clean_symbol = symbol.strip()
        
        # Fetch data
        data = yfu.fetch_ticker_data(
            symbol=clean_symbol,
            start_date=start_date,
            end_date=end_date,
            attempts=3
        )
        
        if data is None or data.empty:
            logger.error(f"Failed to fetch data for {clean_symbol}")
            return False
            
        # Save data
        data_dir = os.path.join(config.DEFAULT_CONFIG['data_directory'], exchange)
        os.makedirs(data_dir, exist_ok=True)
        
        success = yfu.save_ticker_data(clean_symbol, data, data_dir)
        
        if success:
            logger.info(f"Successfully updated {clean_symbol}")
            
        return success
    
    except Exception as e:
        logger.error(f"Error updating {symbol}: {str(e)}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download stock data from Yahoo Finance with improved reliability',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--exchanges', nargs='+', choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX'],
                      help='Specify exchanges to update (e.g., NYSE NASDAQ)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                      help=f'Start date for data retrieval (format: {config.DEFAULT_CONFIG["output_format"]["date_format"]})')
    parser.add_argument('--end-date', type=str,
                      help=f'End date for data retrieval (format: {config.DEFAULT_CONFIG["output_format"]["date_format"]}, default: today)')
    parser.add_argument('--symbol', type=str,
                      help='Update a specific stock symbol instead of an exchange')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of symbols to process in each batch (default: 3)')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of parallel workers (default: 4)')
    parser.add_argument('--allow-problematic', action='store_true',
                      help='Process known problematic symbols (default: skip them)')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode with limited symbols')
    parser.add_argument('--max-symbols', type=int,
                      help='Maximum number of symbols to process (for testing)')
    args = parser.parse_args()
    
    # Validate date format
    date_format = config.DEFAULT_CONFIG['output_format']['date_format']
    try:
        if args.start_date:
            datetime.strptime(args.start_date, date_format)
        if args.end_date:
            datetime.strptime(args.end_date, date_format)
    except ValueError:
        logger.error(f"Invalid date format. Use {date_format}")
        return 1
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup test mode
    max_symbols = None
    if args.test_mode:
        max_symbols = args.max_symbols or 50
        logger.info(f"Running in test mode with max {max_symbols} symbols per exchange")
    
    start_time = time.time()
    
    # Process single symbol
    if args.symbol:
        success = update_single_symbol(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f} seconds")
        return 0 if success else 1
    
    # Process exchanges
    if not args.exchanges:
        logger.error("No exchanges specified. Use --exchanges or --symbol")
        return 1
    
    results = {}
    for exchange in args.exchanges:
        if shutdown_requested:
            logger.info("Shutdown requested, stopping...")
            break
            
        logger.info(f"\nProcessing exchange: {exchange}")
        success = update_exchange(
            exchange=exchange,
            start_date=args.start_date,
            end_date=args.end_date,
            batch_size=args.batch_size,
            workers=args.workers,
            skip_problematic=not args.allow_problematic,
            max_symbols=max_symbols
        )
        results[exchange] = success
        
    # Final summary
    success_count = sum(1 for success in results.values() if success)
    elapsed = time.time() - start_time
    
    logger.info("\nFinal Summary:")
    logger.info(f"Processed {len(results)} exchanges")
    logger.info(f"Successfully updated: {success_count}")
    logger.info(f"Failed: {len(results) - success_count}")
    logger.info(f"Total time: {elapsed:.1f} seconds")
    
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
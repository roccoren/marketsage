#!/usr/bin/env python3
"""
Command-line interface for fetching market data using the enhanced market data fetcher.

Example usage:
    # Fetch AAPL data from NASDAQ for last month
    python fetch_market_data.py --market nasdaq --symbol AAPL --start-date 2024-02-01

    # Fetch all NASDAQ symbols for 2023
    python fetch_market_data.py --market nasdaq --start-date 2023-01-01 --end-date 2023-12-31

    # Backfill specific symbol with custom cache directory
    python fetch_market_data.py --market nyse --symbol IBM --start-date 2020-01-01 --cache-dir /path/to/cache
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from enhanced_market_fetcher import EnhancedMarketDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date format. Please use YYYY-MM-DD. Error: {str(e)}"
        )

def validate_market(market: str) -> str:
    """Validate market name is non-empty."""
    if not market:
        raise argparse.ArgumentTypeError("Market name cannot be empty")
    return market.lower()

def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch market data using the enhanced market data fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch AAPL data from NASDAQ for last month
  %(prog)s --market nasdaq --symbol AAPL --start-date 2024-02-01

  # Fetch all NASDAQ symbols for 2023
  %(prog)s --market nasdaq --start-date 2023-01-01 --end-date 2023-12-31

  # Backfill specific symbol with custom cache directory
  %(prog)s --market nyse --symbol IBM --start-date 2020-01-01 --cache-dir /path/to/cache
        """
    )
    
    parser.add_argument(
        "--market",
        type=validate_market,
        required=True,
        help="Market name (e.g., nasdaq, nyse)"
    )
    
    parser.add_argument(
        "--symbol",
        help="Specific symbol to fetch (optional, fetches all market symbols if not specified)"
    )
    
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default="2020-01-01",
        help="Start date in YYYY-MM-DD format (default: 2020-01-01)"
    )
    
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for storing market data (default: data)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory for cache storage (default: .cache)"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="/datadrive/workspaces/stockscan-market-fetch/config/market",
        help="Directory containing market listings (default: /datadrive/workspaces/stockscan-market-fetch/config/market)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

async def fetch_data(args: argparse.Namespace) -> int:
    """
    Fetch market data based on command line arguments.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        # Set up fetcher
        fetcher = EnhancedMarketDataFetcher(
            data_dir=args.data_dir,
            config_dir=args.config_dir
        )
        
        # Validate market
        if args.market not in fetcher.registry.get_markets():
            logger.error(f"Invalid market: {args.market}")
            return 1
        
        # Validate symbol if provided
        if args.symbol and not fetcher.registry.validate_symbol(args.market, args.symbol):
            logger.error(f"Invalid symbol {args.symbol} for market {args.market}")
            return 1
        
        # Set up dates
        end_date = args.end_date or datetime.now()
        if args.start_date > end_date:
            logger.error("Start date cannot be after end date")
            return 1
        
        # Fetch data
        if args.symbol:
            logger.info(f"Fetching data for {args.symbol} from {args.market}")
            await fetcher.backfill_data(
                args.market,
                args.symbol,
                args.start_date,
                end_date
            )
        else:
            logger.info(f"Fetching all symbols from {args.market}")
            await fetcher.fetch_market_data(
                args.market,
                args.start_date,
                end_date
            )
        
        logger.info("Data fetching completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error information:")
        return 1

def main() -> int:
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create directories if they don't exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.config_dir).mkdir(parents=True, exist_ok=True)
    
    # Run async fetch operation
    try:
        return asyncio.run(fetch_data(args))
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error information:")
        return 1

if __name__ == "__main__":
    sys.exit(main())
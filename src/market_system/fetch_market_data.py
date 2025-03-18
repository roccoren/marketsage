"""Standalone market data fetcher with async processing and rate limiting.
Includes market cap and fundamental data (sector, industry, company name, etc.) in the saved JSON files.
"""
import asyncio
import argparse
from datetime import datetime, timedelta
import logging
import sys
from typing import List, Optional
from pathlib import Path

from src.market_data.providers.enhanced_fetcher import EnhancedMarketFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date format: {e}")

def validate_symbols(symbols_str: str) -> List[str]:
    """Validate and parse symbols string."""
    if not symbols_str:
        return []
    return [s.strip() for s in symbols_str.split(',') if s.strip()]

def get_default_dates():
    """Get default start and end dates."""
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)  # Default start date
    return start_date, end_date

async def fetch_data(args):
    """Fetch market data based on CLI arguments."""
    try:
        # Initialize fetcher with enhanced configuration
        fetcher = EnhancedMarketFetcher(
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            requests_per_minute=args.requests_per_minute,
            cache_ttl_days=1
        )

        # Validate market
        available_markets = fetcher.get_available_markets()
        if args.market and args.market not in available_markets:
            logger.error(f"Invalid market: {args.market}")
            logger.info(f"Available markets: {', '.join(available_markets)}")
            return 1

        # Set up dates
        start_date, end_date = get_default_dates()
        if args.start_date:
            start_date = args.start_date
        if args.end_date:
            end_date = args.end_date

        # Validate dates
        if start_date > end_date:
            logger.error("Start date cannot be after end date")
            return 1

        # Process each specified market
        markets_to_process = [args.market] if args.market else available_markets
        total_markets = len(markets_to_process)

        for idx, market in enumerate(markets_to_process, 1):
            logger.info(f"\nProcessing market {idx}/{total_markets}: {market.upper()}")
            
            # Get and validate symbols
            symbols = None
            if args.symbols:
                symbols = validate_symbols(args.symbols)
                valid_market_symbols = fetcher.get_market_symbols(market)
                invalid_symbols = [s for s in symbols if s not in valid_market_symbols]
                if invalid_symbols:
                    logger.warning(f"Skipping invalid symbols for {market}: {', '.join(invalid_symbols)}")
                symbols = [s for s in symbols if s in valid_market_symbols]
                if not symbols:
                    logger.warning(f"No valid symbols provided for {market}")
                    continue

            try:
                await fetcher.fetch_market_data(
                    market=market,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    chunk_size=args.batch_size
                )
            except Exception as e:
                logger.error(f"Error fetching {market} data: {str(e)}")
                if not args.continue_on_error:
                    return 1

        logger.info("\nData fetching completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        return 1

def main():
    """CLI entry point for market data fetcher."""
    parser = argparse.ArgumentParser(description="Fetch market data with enhanced features including market cap and fundamental data")
    
    # Market selection
    parser.add_argument('-m', '--market', type=str,
                      help='Specific market to fetch (e.g., nyse, nasdaq)')
    parser.add_argument('-s', '--symbols', type=str,
                      help='Comma-separated list of symbols to fetch')
    
    # Date range
    parser.add_argument('--start-date', type=parse_date,
                      help='Start date (YYYY-MM-DD format, default: 2020-01-01)')
    parser.add_argument('--end-date', type=parse_date,
                      help='End date (YYYY-MM-DD format, default: today)')
    
    # Configuration
    parser.add_argument('--data-dir', type=str, default="data",
                      help='Directory for storing data')
    parser.add_argument('--config-dir', type=str, default="config/market",
                      help='Directory containing market configuration files')
    
    # Performance tuning
    parser.add_argument('--max-concurrent', type=int, default=5,
                      help='Maximum concurrent downloads')
    parser.add_argument('--batch-size', type=int, default=20,
                      help='Number of symbols to process in each batch')
    parser.add_argument('--requests-per-minute', type=int, default=60,
                      help='Maximum API requests per minute')
    
    # Error handling
    parser.add_argument('--continue-on-error', action='store_true',
                      help='Continue processing if a market fails')

    args = parser.parse_args()

    # Ensure data directory exists
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Run async fetcher
    return asyncio.run(fetch_data(args))

if __name__ == "__main__":
    sys.exit(main())

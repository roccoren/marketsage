#!/usr/bin/env python3

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import tqdm
from .enhanced_market_fetcher import EnhancedMarketDataFetcher

# Configure logging
def setup_logging(log_file: Optional[Path] = None, verbose: bool = False):
    """Configure logging to both file and console with rotation."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger()
    
    if log_file:
        # Add rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger

class MarketDataCLI:
    def __init__(self, args):
        self.args = args
        self.fetcher = EnhancedMarketDataFetcher()
        self.logger = setup_logging(
            log_file=Path(args.output_dir) / "market_data.log" if args.output_dir else None,
            verbose=args.verbose
        )

    async def process_market(self):
        """Process market data with progress tracking and error handling."""
        try:
            # Get symbols for the market and filter out special securities
            all_symbols = self.fetcher.registry.get_symbols(self.args.market.lower())
            if not all_symbols:
                self.logger.error(f"No symbols found for market: {self.args.market}")
                return
            
            # Filter out preferred shares, warrants, and special securities
            symbols = {symbol for symbol in all_symbols
                      if '$' not in symbol
                      and '.' not in symbol
                      and '-' not in symbol}
            
            self.logger.info(f"Processing {len(symbols)} regular symbols out of {len(all_symbols)} total for {self.args.market}")
            
            # Convert set to list and create batches
            symbols_list = list(symbols)
            batches = [symbols_list[i:i + self.args.batch_size]
                      for i in range(0, len(symbols_list), self.args.batch_size)]
            
            # Process each batch with progress bar
            with tqdm.tqdm(total=len(symbols), desc="Processing symbols") as pbar:
                for batch in batches:
                    try:
                        # Process batch
                        tasks = [
                            self.process_symbol(symbol)
                            for symbol in batch
                        ]
                        await asyncio.gather(*tasks)
                        
                        # Update progress
                        pbar.update(len(batch))
                        
                        # Always apply rate limiting between batches with minimum delay
                        await asyncio.sleep(max(5.0, self.args.delay))
                            
                    except Exception as e:
                        self.logger.error(f"Error processing batch: {str(e)}")
                        self.logger.info("Waiting 60 seconds before next batch...")
                        await asyncio.sleep(60)  # Longer delay after error
                        continue
        except Exception as e:
            self.logger.error(f"Error in process_market: {str(e)}")

    async def process_symbol(self, symbol: str, max_retries: int = 3):
        """Process individual symbol with error handling, gap detection, and retries."""
        retries = 0
        while retries < max_retries:
            try:
                # Check existing data and determine what needs to be fetched
                await self.fetcher.backfill_data(
                    self.args.market.lower(),
                    symbol,
                    self.args.start_date
                )
                return  # Success, exit retry loop
                
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    self.logger.warning(f"Retry {retries}/{max_retries} for {symbol}: {str(e)}")
                    await asyncio.sleep(30 * retries)  # Exponential backoff
                else:
                    self.logger.error(f"Failed {symbol} after {max_retries} retries: {str(e)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Market Data CLI - Download and manage stock market data"
    )
    
    # Required arguments
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--market",
        type=str,
        required=True,
        choices=['NYSE', 'NASDAQ'],
        help="Market name (NYSE/NASDAQ)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for API calls (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between batches in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory path (default: data/)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Ensure output directory exists
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run CLI
    cli = MarketDataCLI(args)
    await cli.process_market()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
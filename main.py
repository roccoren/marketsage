#!/usr/bin/env python3
import os
import json
import logging
import tempfile
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List

from utils.yfinance_utils import download_with_session
from utils.async_utils import YahooFinanceClient, UpdateProgressTracker
import update_stocks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockQuoteManager:
    def __init__(self):
        self.base_dir = Path("data")
        self.exchanges = {
            "NYSE": self.base_dir / "NYSE",
            "NASDAQ": self.base_dir / "NASDAQ",
            "HKEX": self.base_dir / "HKEX",
            "ASX": self.base_dir / "ASX"
        }
        self.client = YahooFinanceClient(rate_limit_ms=10000, batch_size=3)
        self.progress_tracker = UpdateProgressTracker("data")
        self._create_directories()

    def _create_directories(self) -> None:
        """Create directory structure for each exchange if it doesn't exist."""
        for exchange_dir in self.exchanges.values():
            exchange_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {exchange_dir}")

    def _validate_quote_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the stock quote data.
        
        Args:
            data: DataFrame containing stock quote data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False
            
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Found: {data.columns}")
            return False
            
        if data.isnull().values.any():
            logger.warning("Data contains null values")
            return False
            
        return True

    def _prepare_quote_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare quote data for storage.
        
        Args:
            data: DataFrame containing latest quote
            
        Returns:
            dict: Formatted quote data
        """
        latest_row = data.iloc[-1]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "trading_date": data.index[-1].isoformat(),
            "price": {
                "open": float(latest_row["Open"]),
                "high": float(latest_row["High"]),
                "low": float(latest_row["Low"]),
                "close": float(latest_row["Close"])
            },
            "volume": int(latest_row["Volume"]),
            "additional_metrics": {
                "vwap": float(latest_row.get("VWAP", 0)) if "VWAP" in latest_row else None,
                "typical_price": (float(latest_row["High"]) + float(latest_row["Low"]) + float(latest_row["Close"])) / 3
            }
        }

    def _atomic_write_json(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """
        Atomically write data to JSON file using a temporary file.
        
        Args:
            filepath: Path to the target JSON file
            data: Data to write
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        try:
            # Create a temporary file in the same directory
            with tempfile.NamedTemporaryFile(mode='w', 
                                           dir=filepath.parent,
                                           delete=False,
                                           suffix='.tmp') as tmp_file:
                # Write data to temporary file
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                
            # Atomic rename
            os.rename(tmp_file.name, filepath)
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {filepath}: {str(e)}")
            # Clean up temporary file if it exists
            if 'tmp_file' in locals():
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
            return False

    def update_stock_quote(self, symbol: str, exchange: str) -> bool:
        """
        Fetch and update quote data for a single stock.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code (NYSE, NASDAQ, HKEX, ASX)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if exchange not in self.exchanges:
            logger.error(f"Invalid exchange: {exchange}")
            return False
            
        try:
            # Fetch latest quote data
            data = download_with_session(symbol, period="1d", interval="1m")
            
            if not self._validate_quote_data(data):
                logger.error(f"Invalid quote data for {symbol}")
                return False
                
            # Prepare quote data
            quote_data = self._prepare_quote_data(data)
            
            # Construct file path
            file_path = self.exchanges[exchange] / f"{symbol}.json"
            
            # Atomic write
            if self._atomic_write_json(file_path, quote_data):
                logger.info(f"Successfully updated quote for {symbol} ({exchange})")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating quote for {symbol}: {str(e)}")
            return False

    def get_yahoo_symbol(self, exchange: str, symbol: str) -> str:
        """Get Yahoo Finance formatted symbol."""
        # if exchange == 'ASX':
        #     return f"{symbol}.AX"
        # elif exchange == 'HKEX':
        #     return f"{symbol}.HK"
        return symbol

    async def should_update_historical(self, symbol_path: Path, start_date: str) -> bool:
        """Check if historical data needs updating."""
        try:
            if not symbol_path.exists():
                return True
                
            with open(symbol_path) as f:
                data = json.load(f)
                
            timestamps = data.get('chart', {}).get('result', [{}])[0].get('timestamp', [])
            if not timestamps:
                return True
                
            latest_date = datetime.fromtimestamp(timestamps[-1]).date()
            today = datetime.now().date()
            
            return (today - latest_date).days > 1
                
        except Exception as e:
            logger.error(f"Error checking {symbol_path}: {e}")
            return True

    async def fetch_historical_data(
        self,
        exchange: str,
        start_date: Optional[str] = None
    ) -> None:
        """
        Fetch historical data for all symbols in an exchange.
        
        Args:
            exchange: Exchange to process
            start_date: Start date for historical data (YYYY-MM-DD)
        """
        if exchange not in self.exchanges:
            logger.error(f"Invalid exchange: {exchange}")
            return

        # Load exchange symbols
        listing_file = self.base_dir / f"{exchange.lower()}_filtered_listings.json"
        try:
            with open(listing_file) as f:
                listings = json.load(f)
        except Exception as e:
            logger.error(f"Error loading {exchange} listings: {e}")
            return

        # Initialize progress tracking
        await self.progress_tracker.load_progress()
        total_symbols = len(listings)
        completed = 0

        async with self.client:
            for listing in listings:
                symbol = listing['symbol']
                yahoo_symbol = self.get_yahoo_symbol(exchange, symbol)
                output_path = self.exchanges[exchange] / f"{symbol}.json"

                if not await self.should_update_historical(output_path, start_date):
                    logger.debug(f"Skipping {symbol} - data is current")
                    completed += 1
                    continue

                try:
                    data = await self.client.get_history(
                        yahoo_symbol,
                        start_date=start_date,
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    if data:
                        if self._atomic_write_json(output_path, data):
                            completed += 1
                            logger.info(f"Updated historical data for {symbol}")
                        else:
                            await self.progress_tracker.add_failed_symbol(exchange, symbol)
                    else:
                        logger.warning(f"No data returned for {symbol}")
                        await self.progress_tracker.add_failed_symbol(exchange, symbol)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    await self.progress_tracker.add_failed_symbol(exchange, symbol)

                await self.progress_tracker.update_exchange_progress(
                    exchange,
                    total_symbols,
                    completed
                )

        logger.info(f"Completed {exchange}: {completed}/{total_symbols} symbols processed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stock Data Manager')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Update listings command
    update_parser = subparsers.add_parser('update-listings', help='Update stock listings')
    update_parser.add_argument('--exchanges', nargs='+', 
                             choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                             help='Specific exchanges to update')

    # Update quotes command
    quotes_parser = subparsers.add_parser('update-quotes', help='Update latest quotes')
    quotes_parser.add_argument('--exchanges', nargs='+',
                             choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                             help='Specific exchanges to update')

    # Historical data command
    historical_parser = subparsers.add_parser('fetch-historical', 
                                            help='Fetch historical data')
    historical_parser.add_argument('--exchanges', nargs='+',
                                 choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                                 help='Specific exchanges to process')
    historical_parser.add_argument('--start-date', default='2024-06-01',
                                 help='Start date for historical data (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.command == 'update-listings':
        update_stocks.update_all_listings(args.exchanges)
    else:
        manager = StockQuoteManager()

        if args.command == 'update-quotes':
            # Process each selected exchange
            for exchange in (args.exchanges or manager.exchanges.keys()):
                try:
                    listing_file = Path(f"data/{exchange.lower()}_filtered_listings.json")
                    if listing_file.exists():
                        with open(listing_file, 'r') as f:
                            listings = json.load(f)
                            
                        for listing in listings:
                            manager.update_stock_quote(listing["symbol"], exchange)
                except Exception as e:
                    logger.error(f"Error processing {exchange}: {str(e)}")

        elif args.command == 'fetch-historical':
            # Process historical data for selected exchanges
            for exchange in (args.exchanges or manager.exchanges.keys()):
                asyncio.run(manager.fetch_historical_data(
                    exchange,
                    start_date=args.start_date
                ))

if __name__ == "__main__":
    main()
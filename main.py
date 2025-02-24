#!/usr/bin/env python3
import os
import json
import logging
import random
import tempfile
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List

from utils.yfinance_utils import download_with_session
from strategies.macd import MACDStrategy
from utils.async_utils import YahooFinanceClient, UpdateProgressTracker
import update_stocks

class DebugConfig:
    """Manages component-level debug settings."""
    
    def __init__(self):
        self.components = {
            'macd': False,
            'quotes': False,
            'historical': False,
            'listings': False
        }
        
    def enable_component(self, component: str) -> None:
        """Enable debug logging for a specific component."""
        if component in self.components:
            self.components[component] = True
            
    def is_enabled(self, component: str) -> bool:
        """Check if debug logging is enabled for a component."""
        return self.components.get(component, False)
        
    @staticmethod
    def setup_logging():
        """Configure basic logging setup."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True,
            handlers=[
                logging.FileHandler('stock_update.log'),
                logging.StreamHandler()
            ]
        )

debug_config = DebugConfig()
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
            
    def log_symbol_info(self, exchange: str, symbol: str, component: str) -> None:
        """Log symbol processing information if debug is enabled for component."""
        if debug_config.is_enabled(component):
            logger.debug(f"\nDebug info for {exchange}:{symbol}")
            file_path = self.exchanges[exchange] / f"{symbol}.json"
            logger.debug(f"File path: {file_path}")
            logger.debug(f"File exists: {file_path.exists()}")
            logger.debug(f"Yahoo symbol: {self.get_yahoo_symbol(exchange, symbol)}")
            
    def log_debug(self, component: str, message: str) -> None:
        """Log debug message if enabled for component."""
        if debug_config.is_enabled(component):
            logger.debug(message)

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

    def analyze_macd_signals(self, exchange: str, symbol: str) -> Optional[List[Dict]]:
        """
        Analyze historical data for MACD signals.
        
        Args:
            exchange: Exchange code (NYSE, NASDAQ, HKEX, ASX)
            symbol: Stock symbol
            
        Returns:
            List of MACD signals if found, None if error occurs
        """
        try:
            file_path = self.exchanges[exchange] / f"{symbol}.json"
            if not file_path.exists():
                logger.error(f"No historical data found for {symbol}")
                return None

            with open(file_path) as f:
                self.log_debug('macd', f"Reading data for {symbol}")
                data = json.load(f)
                
            # Detailed debug logging
            if debug_config.is_enabled('macd'):
                self.log_debug('macd', f"Data structure:")
                self.log_debug('macd', f"Top level keys: {list(data.keys())}")
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    self.log_debug('macd', f"Chart keys: {list(result.keys())}")
            
            # Extract raw data
            quote_data = data['chart']['result'][0]['indicators']['quote'][0]
            
            # Create DataFrame with explicitly named columns
            df = pd.DataFrame({
                'Close': quote_data['close'],
                'Open': quote_data['open'],
                'High': quote_data['high'],
                'Low': quote_data['low'],
                'Volume': quote_data['volume']
            })
            df.columns = df.columns.tolist()  # Convert numpy array to list of strings
            df.index = pd.to_datetime([datetime.fromtimestamp(x) for x in data['chart']['result'][0]['timestamp']])
            
            logger.info("-" * 80)
            logger.info(f"DataFrame for {symbol}:")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns type: {type(df.columns)}")
            logger.info(f"Column 'Close' type: {type(df['Close'])}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"First few rows:\n{df.head()}")
            logger.info("-" * 80)
            
            result = MACDStrategy().analyze(df)
            return result
        except Exception as e:
            logger.error(f"Error analyzing MACD signals for {symbol}: {str(e)}\nDataFrame state: {df.columns if 'df' in locals() else 'DataFrame not created'}")
            return None

    def analyze_macd_for_last_7_days(self, exchange: str, symbol: str) -> Optional[Dict]:
        """
        Analyze historical data for MACD signals for the last 7 trading days and format the output.

        Args:
            exchange: Exchange code (NYSE, NASDAQ, HKEX, ASX)
            symbol: Stock symbol

        Returns:
            Dictionary containing formatted MACD signals for the last 7 days, or None if error occurs
        """
        signals = self.analyze_macd_signals(exchange, symbol)
        if not signals:
            logger.info(f"No MACD signals found for {symbol} in the last 7 days.")
            return None

        formatted_output = {
            "symbol": symbol,
            "exchange": exchange,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "signals": []
        }

        for signal in signals:
            signal_data = {
                "signal_date": signal['SignalDate'],
                "type": signal['Type'],
                "price": signal['Price'],
                "macd_value": signal['MACD'],
                "signal_value": signal['Signal'],
                "recommendation": signal['Type']
            }
            formatted_output["signals"].append(signal_data)

        return formatted_output
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
        if exchange == 'ASX':
            return f"{symbol}.AX"
        elif exchange == 'HKEX':
            # Remove .HK if it exists, then add it back
            base_symbol = symbol.replace('.HK', '')
            return f"{base_symbol}.HK"
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
                    self.log_debug('historical', f"Skipping {symbol} - data is current")
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

    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument('--debug-macd', action='store_true', help='Enable MACD analysis debug logging')
    debug_group.add_argument('--debug-quotes', action='store_true', help='Enable quote updates debug logging')
    debug_group.add_argument('--debug-historical', action='store_true', help='Enable historical data debug logging')
    debug_group.add_argument('--debug-listings', action='store_true', help='Enable listings debug logging')

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

    # MACD analysis command
    macd_parser = subparsers.add_parser('analyze-macd',
                                      help='Analyze stocks using MACD strategy')
    macd_parser.add_argument('--exchanges', nargs='+',
                          choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                          help='Specific exchanges to analyze')
    macd_parser.add_argument('--output-dir', default='output', help='Output directory for MACD signals')
    macd_parser.add_argument('--output', help='Output file for MACD signals')

    args = parser.parse_args()
    
    # Setup logging and debug configuration
    DebugConfig.setup_logging()
    
    # Configure component-level debugging
    if args.debug_macd: debug_config.enable_component('macd')
    if args.debug_quotes: debug_config.enable_component('quotes')
    if args.debug_historical: debug_config.enable_component('historical')
    if args.debug_listings: debug_config.enable_component('listings')

    if args.command == 'update-listings':
        update_stocks.update_all_listings(args.exchanges)
    else:
        manager = StockQuoteManager()

        if args.command == 'update-quotes':
            # Process each selected exchange
            for exchange in (args.exchanges or manager.exchanges.keys()):
                logger.info(f"\n{'='*50}\nProcessing exchange: {exchange}\n{'='*50}")
                try:
                    logger.debug(f"Looking for listings file: data/{exchange.lower()}_filtered_listings.json")
                    listing_file = Path(f"data/{exchange.lower()}_filtered_listings.json")
                    if listing_file.exists():
                        with open(listing_file, 'r') as f:
                            logger.debug("Loading listings file...")
                            listings = json.load(f)
                            logger.info(f"Found {len(listings)} symbols, first 5: {[l['symbol'] for l in listings[:5]]}")
                        logger.info(f"Loaded {len(listings)} symbols from {listing_file}")
                        if listings:
                            logger.info(f"Sample symbols: {[l['symbol'] for l in listings[:5]]}")
                            
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
        
        # MACD Analysis
        elif args.command == 'analyze-macd':
            signals = []
            manager.log_symbol_info('HKEX', '0001.HK', 'macd')  # Test with a known HKEX symbol
            current_date = datetime.now().strftime('%Y%m%d')
            logger.info("Current strategy name (case-sensitive): MACD")
            
            def generate_random_suffix():
                """Generate 4-character random alphanumeric suffix."""
                chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                return ''.join(random.choices(chars, k=4))
                
            # Create output directory if it doesn't exist
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            all_signals = []
            for exchange in (args.exchanges or manager.exchanges.keys()):
                try:
                    exchange_signals = []
                    listing_file = Path(f"data/{exchange.lower()}_filtered_listings.json")
                    if listing_file.exists():
                        with open(listing_file, 'r') as f:
                            logger.info(f"Reading listings from {listing_file}")
                            listings = json.load(f)
                            manager.log_debug('macd', f"HKEX listings loaded: {len(listings)} symbols found")
                            logger.info(f"Processing {len(listings)} symbols for {exchange}")

                            for listing in listings:
                                manager.log_debug('macd', f"Processing listing: {listing}")
                                symbol = listing["symbol"]
                                manager.log_debug('macd', f"Attempting to analyze MACD for {exchange}:{symbol}")
                                macd_analysis_result = manager.analyze_macd_for_last_7_days(exchange, symbol)
                                manager.log_debug('macd', f"MACD analysis result for {symbol}: {macd_analysis_result}")
                                if macd_analysis_result:
                                    logger.info(f"✓ Successfully analyzed {symbol}: Found MACD signals")
                                    exchange_signals.append(macd_analysis_result)
                                    signals.extend(exchange_signals)
                                    print(f"\nMACD Signals for {exchange} - {symbol}:")
                                    if macd_analysis_result and macd_analysis_result['signals']:
                                        for signal in macd_analysis_result['signals']:
                                            print(f"  Date: {signal['signal_date']}, Signal: {signal['type']}, Price: {signal['price']:.2f}")
                                    else:
                                        print(f"  No MACD signals found for this exchange in the last 7 days.")

                except Exception as e:
                    logger.warning(f"Non-critical error while processing {exchange} - continuing with analysis: {str(e)}")
                    logger.debug(f"Exception details for {exchange}: {type(e).__name__}: {str(e)}")
                    logger.debug(f"Exchange signals found before error: {len(exchange_signals) if 'exchange_signals' in locals() else 0}")
            
            # Write all signals to a single combined file
            # Using format: {market}_{strategy}_{date}_{id}.json
            market = "MULTI" if len(args.exchanges or manager.exchanges.keys()) > 1 else next(iter(args.exchanges or manager.exchanges.keys()))
            output_path = output_dir / f"{market}_MACD_{current_date}_{generate_random_suffix()}.json"
            with open(output_path, 'w') as f:
                json.dump(signals, f, indent=2)
            logger.info(f"Saved all MACD signals to {output_path}")
            
            # Log analysis completion
            logger.info(f"Total signals collected across all exchanges: {len(signals)}")
            logger.info(f"Signals list content: {signals}")
            if not signals:
                logger.info("\n📊 Analysis complete: No MACD signals found across any exchanges")
            else:
                logger.info("\n📊 Analysis Summary:")
                logger.info(f"✓ Found {len(signals)} MACD signals across all exchanges")
                logger.info("✓ Individual exchange signal files have been generated")
                logger.info("✓ Check the output directory for detailed results")

if __name__ == "__main__":
    main()
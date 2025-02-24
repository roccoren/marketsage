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
from strategies.combined_signal import CombinedSignalStrategy
from utils.async_utils import YahooFinanceClient, UpdateProgressTracker
import update_stocks

class DebugConfig:
    """Manages component-level debug settings."""
    
    def __init__(self):
        self.components = {
            'macd': False,
            'quotes': False,
            'historical': False,
            'listings': False,
            'combined_signal': False
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
        """Validate the stock quote data."""
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
        """Prepare quote data for storage."""
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

    def update_stock_quote(self, symbol: str, exchange: str) -> bool:
        """Fetch and update quote data for a single stock."""
        if exchange not in self.exchanges:
            logger.error(f"Invalid exchange: {exchange}")
            return False
            
        try:
            data = download_with_session(symbol, period="1d", interval="1m")
            
            if not self._validate_quote_data(data):
                logger.error(f"Invalid quote data for {symbol}")
                return False
                
            quote_data = self._prepare_quote_data(data)
            file_path = self.exchanges[exchange] / f"{symbol}.json"
            
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
            if not symbol.endswith('.AX'):
                logger.debug(f"Formatting ASX symbol: {symbol} -> {symbol}.AX")
                return f"{symbol}.AX"
            logger.debug(f"Symbol {symbol} already has .AX suffix")
            return symbol
        elif exchange == 'HKEX':
            if not symbol.endswith('.HK'):
                base_symbol = symbol.replace('.HK', '')
                logger.debug(f"Formatting HKEX symbol: {symbol} -> {base_symbol}.HK")
                return f"{base_symbol}.HK"
            logger.debug(f"Symbol {symbol} already has .HK suffix")
            return symbol
        return symbol

    def _atomic_write_json(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Atomically write data to JSON file."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', 
                                         dir=filepath.parent,
                                         delete=False,
                                         suffix='.tmp') as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                
            os.rename(tmp_file.name, filepath)
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {filepath}: {str(e)}")
            if 'tmp_file' in locals():
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
            return False

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
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None
    ) -> None:
        """Fetch historical data for all symbols in an exchange."""
        if exchange not in self.exchanges:
            logger.error(f"Invalid exchange: {exchange}")
            return

        # Load filtered listings
        try:
            listings_file = Path(f"data/{exchange.lower()}_filtered_listings.json")
            if not listings_file.exists():
                logger.error(f"Filtered listings file not found: {listings_file}")
                return
                
            with open(listings_file) as f:
                filtered_listings = json.load(f)
                valid_symbols = {item['symbol'] for item in filtered_listings}
                logger.info(f"Loaded {len(valid_symbols)} filtered symbols for {exchange}")
        except Exception as e:
            logger.error(f"Error loading filtered listings for {exchange}: {e}")
            return

        if symbols is None:
            symbols = list(valid_symbols)
        else:
            symbols = list(set(symbols) & valid_symbols)

        await self.progress_tracker.load_progress()
        total_symbols = len(symbols)
        completed = 0
        failed_symbols = []
        rate_limit_errors = 0
        max_rate_limit_retries = 3

        async with self.client:
            for symbol in symbols:
                yahoo_symbol = self.get_yahoo_symbol(exchange, symbol)
                output_path = self.exchanges[exchange] / f"{symbol}.json"
                
                if rate_limit_errors >= max_rate_limit_retries:
                    logger.error(f"Too many rate limit errors, pausing {exchange} processing")
                    break

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
                        logger.debug(f"Retrieved data for {symbol}: {len(data.get('chart', {}).get('result', []))}")
                        if self._atomic_write_json(output_path, data):
                            completed += 1
                            logger.info(f"Updated historical data for {symbol}")
                        else:
                            logger.error(f"Failed to write data for {symbol}")
                            if symbol not in failed_symbols:
                                failed_symbols.append(symbol)
                            await self.progress_tracker.add_failed_symbol(exchange, symbol)
                    else:
                        logger.warning(f"No data returned for {symbol}")
                        await self.progress_tracker.add_failed_symbol(exchange, symbol)

                    # Transform data structure for analysis compatibility
                    if data and 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        if 'timestamp' in result and 'indicators' in result:
                            quote = result['indicators']['quote'][0]
                            timestamps = result['timestamp']
                            
                            # Create history entries
                            history = []
                            for i in range(len(timestamps)):
                                if (quote['open'][i] is not None and
                                    quote['high'][i] is not None and
                                    quote['low'][i] is not None and
                                    quote['close'][i] is not None and
                                    quote['volume'][i] is not None):
                                    history_entry = {
                                        'date': datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d'),
                                        'open': quote['open'][i],
                                        'high': quote['high'][i],
                                        'low': quote['low'][i],
                                        'close': quote['close'][i],
                                        'volume': quote['volume'][i]
                                    }
                                    history.append(history_entry)
                            
                            # Replace the original data structure with the new format
                            transformed_data = {'history': history}
                            if self._atomic_write_json(output_path, transformed_data):
                                logger.info(f"Successfully transformed historical data for {symbol}")
                            else:
                                logger.error(f"Failed to write transformed data for {symbol}")
                                await self.progress_tracker.add_failed_symbol(exchange, symbol)
                    else:
                        logger.error(f"Invalid data structure for {symbol}")
                        await self.progress_tracker.add_failed_symbol(exchange, symbol)

                except Exception as e:
                    if "rate limit" in str(e).lower():
                        rate_limit_errors += 1
                        logger.warning(f"Rate limit error for {symbol}, waiting longer...")
                        await asyncio.sleep(30)  # Extended pause on rate limit
                    else:
                        logger.error(f"Error processing {symbol}: {e}")
                    if symbol not in failed_symbols:
                        failed_symbols.append(symbol)
                    await self.progress_tracker.add_failed_symbol(exchange, symbol)

                await self.progress_tracker.update_exchange_progress(
                    exchange,
                    total_symbols,
                    completed
                )

        # Final summary
        logger.info(f"\nExchange {exchange} Summary:")
        logger.info(f"Total symbols: {total_symbols}")
        logger.info(f"Successfully processed: {completed}")
        logger.info(f"Failed symbols: {len(failed_symbols)}")
        if failed_symbols:
            logger.info(f"Failed symbol list: {failed_symbols}")
        if rate_limit_errors > 0:
            logger.warning(f"Rate limit errors encountered: {rate_limit_errors}")

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

    # MACD analysis command
    macd_parser = subparsers.add_parser('analyze-macd',
                                      help='Analyze stocks using MACD strategy')
    macd_parser.add_argument('--exchanges', nargs='+',
                           choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                           help='Specific exchanges to analyze')
    macd_parser.add_argument('--output-dir', default='output', help='Output directory for MACD signals')
    macd_parser.add_argument('--output', help='Output file for MACD signals')
    macd_parser.add_argument('--debug', action='store_true', help='Enable MACD debug logging')

    # Add debug options to each subparser
    for p in [update_parser, quotes_parser, historical_parser, macd_parser]:
        debug_group = p.add_argument_group('Debug Options')
        debug_group.add_argument('--debug-quotes', action='store_true', help='Enable quote updates debug logging')
        debug_group.add_argument('--debug-historical', action='store_true', help='Enable historical data debug logging')

    # Combined strategy analysis command
    combined_parser = subparsers.add_parser('analyze-combined',
                                         help='Analyze stocks using combined strategy (Momentum, BB, MACD)')
    combined_parser.add_argument('--exchanges', nargs='+',
                               choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                               help='Specific exchanges to analyze')
    combined_parser.add_argument('--output-dir', default='output', help='Output directory for combined signals')
    combined_parser.add_argument('--output', help='Output file for combined signals')
    combined_parser.add_argument('--debug', action='store_true', help='Enable combined strategy debug logging')

    args = parser.parse_args()
    
    # Setup logging and debug configuration
    DebugConfig.setup_logging()
    
    # Configure component-level debugging
    if hasattr(args, 'debug_quotes') and args.debug_quotes: debug_config.enable_component('quotes')
    if hasattr(args, 'debug_historical') and args.debug_historical: debug_config.enable_component('historical')
    
    if args.command == 'analyze-macd' and args.debug: debug_config.enable_component('macd')
    elif args.command == 'analyze-combined' and args.debug: debug_config.enable_component('combined_signal')

    if args.command == 'update-listings':
        update_stocks.update_all_listings(args.exchanges)
    else:
        manager = StockQuoteManager()

        if args.command == 'update-quotes':
            # Process each selected exchange
            for exchange in (args.exchanges or manager.exchanges.keys()):
                try:
                    logger.info(f"\n{'='*50}\nProcessing exchange: {exchange}\n{'='*50}")
                    exchange_dir = manager.exchanges[exchange]
                    logger.info(f"Scanning directory: {exchange_dir}")
                    symbol_files = list(exchange_dir.glob('*.json'))
                    logger.info(f"Found {len(symbol_files)} existing symbol files")
                    if symbol_files:
                        sample_symbols = [f.stem for f in symbol_files[:5]]
                        logger.info(f"Sample symbols: {sample_symbols}")
                        
                    for symbol_file in symbol_files:
                        symbol = symbol_file.stem  # Get filename without .json extension
                        manager.update_stock_quote(symbol, exchange)
                except Exception as e:
                    logger.error(f"Error processing {exchange}: {str(e)}")

        elif args.command == 'fetch-historical':
            # Process historical data for selected exchanges
            for exchange in (args.exchanges or manager.exchanges.keys()):
                asyncio.run(manager.fetch_historical_data(
                    exchange,
                    start_date=args.start_date
                ))
        
        elif args.command == 'analyze-combined':
            signals = []
            current_date = datetime.now().strftime('%Y%m%d')
            logger.info("\nCurrent strategy: Combined (Momentum, BB, MACD)")

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            def generate_random_suffix():
                chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                return ''.join(random.choices(chars, k=4))

            for exchange in (args.exchanges or manager.exchanges.keys()):
                try:
                    exchange_dir = manager.exchanges[exchange]
                    logger.info(f"Scanning directory: {exchange_dir}")
                    symbol_files = list(exchange_dir.glob('*.json'))
                    logger.info(f"Found {len(symbol_files)} symbol files in {exchange}")
                    
                    for symbol_file in symbol_files:
                        symbol = symbol_file.stem
                        logger.info(f"\nProcessing {symbol} with combined strategy")

                        logger.info(f"Attempting to read file: {symbol_file.absolute()}")
                        with open(symbol_file) as f:
                            data = json.load(f)
                            logger.info(f"File content keys: {list(data.keys())}")
                        
                        if 'history' not in data:
                            logger.warning(f"No historical data found in {symbol_file}")
                            continue

                        logger.debug(f"Found historical data for {symbol}")
                        df = pd.DataFrame(data['history'])
                        logger.info(f"Initial data shape: {df.shape}")
                        df['Date'] = pd.to_datetime(df['date'])
                        df.set_index('Date', inplace=True)
                        df.columns = df.columns.str.capitalize()
                        logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
                        
                        strategy = CombinedSignalStrategy()
                        analysis_result = strategy.analyze(df)
                        
                        if analysis_result:
                            logger.info(f"✓ Found signals for {symbol}")
                            for signal in analysis_result:
                                signal['Symbol'] = symbol
                                signal['Exchange'] = exchange
                            signals.extend(analysis_result)
                            print(f"\nCombined Strategy Signals for {exchange} - {symbol}:")
                            for signal in analysis_result:
                                print(f"  Date: {signal['Date']}, Type: {signal['Type']}, Price: {signal['Price']:.2f}")

                except Exception as e:
                    logger.error(f"Error processing {exchange}: {str(e)}")
                    continue
            
            if signals:
                market = "MULTI" if len(args.exchanges or manager.exchanges.keys()) > 1 else next(iter(args.exchanges or manager.exchanges.keys()))
                output_path = output_dir / f"{market}_COMBINED_{current_date}_{generate_random_suffix()}.json"
                with open(output_path, 'w') as f:
                    json.dump(signals, f, indent=2)
                logger.info(f"\n📊 Analysis Summary:")
                logger.info(f"✓ Found {len(signals)} combined signals")
                logger.info(f"✓ Signals saved to {output_path}")
            else:
                logger.info("\n📊 Analysis complete: No signals found")

if __name__ == "__main__":
    main()
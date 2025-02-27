from typing import List, Optional, Dict, Any, Tuple, Iterator, TypedDict, Union
import pandas as pd
import numpy as np
from src.strategies.trading_analysis import TradingAnalysis
import logging
import time
import pytz
from datetime import datetime, timedelta, time as dt_time
import yfinance as yf
from src.utils.yfinance_utils import get_cached_session
from pathlib import Path
import json
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from src.strategies.base import Strategy
from src.utils.async_utils import YahooFinanceClient, CircuitBreaker, UpdateProgressTracker
from src.utils.data_cleaner import clean_historical_data
from src.utils.filename_generator import FilenameGenerator

logger = logging.getLogger(__name__)

class ExchangeConfig(TypedDict):
    name: str
    timezone: str
    market_open: dt_time
    market_close: dt_time
    validation_symbol: str

class StockData(TypedDict):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjClose: float

class SignalData(TypedDict):
    Symbol: str
    Exchange: str
    Timestamp: str
    Strategy: str
    Signal: str
    Confidence: float

class StockScanner:
    class MarketTimes:
        """Helper class for market timing operations."""
        @staticmethod
        def get_trading_day(tz_name: str, 
                          market_open: dt_time, 
                          market_close: dt_time,
                          validation_symbol: Optional[str] = None,
                          check_data: bool = True) -> datetime:
            """Get the most recent trading day for a given market timezone."""
            tz = pytz.timezone(tz_name)
            current_time = datetime.now(tz)
            proposed_trading_day = None

            if current_time.weekday() >= 5:
                days_to_subtract = current_time.weekday() - 4
                last_trading = current_time - timedelta(days=days_to_subtract)
                proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
            elif current_time.time() < market_open:
                last_trading = current_time - timedelta(days=1)
                while last_trading.weekday() >= 5:
                    last_trading -= timedelta(days=1)
                proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
            elif current_time.time() >= market_close:
                proposed_trading_day = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                proposed_trading_day = (current_time - timedelta(days=1)).replace(
                    hour=16, minute=0, second=0, microsecond=0
                )

            if not check_data or not validation_symbol:
                return proposed_trading_day

            try:
                session = get_cached_session()
                ticker = yf.Ticker(validation_symbol, session=session)
                
                if validation_symbol.endswith('.AX'):
                    end_timestamp = int(proposed_trading_day.timestamp())
                    df = ticker.history(period="5d", end=end_timestamp)
                else:
                    end_date = proposed_trading_day.astimezone(pytz.UTC).strftime('%Y-%m-%d')
                    df = ticker.history(period="5d", end=end_date)
                
                if df.empty:
                    logger.warning(f"No recent market data available to verify trading day for {validation_symbol}")
                    return proposed_trading_day
                    
                last_actual_trading = df.index[-1].to_pydatetime().astimezone(tz)
                return last_actual_trading.replace(hour=16, minute=0, second=0, microsecond=0)
                
            except Exception as e:
                logger.error(f"Error verifying trading day against market data: {str(e)}")
                return proposed_trading_day

    class ResultsManager:
        """Manages storage and retrieval of scan results."""
        BATCH_SIZE = 1000  # Number of results to process at once
        
        REQUIRED_FIELDS = {
            'Symbol': str,
            'Exchange': str,
            'Timestamp': str,
            'Strategy': str
        }
        
        def __init__(self, output_dir: Path, exchange: str, strategies: List[Strategy]):
            self.output_dir = output_dir
            
            # Generate filename using strategy names
            generator = FilenameGenerator()
            strategy_names = [s.name.lower().replace(' ', '_').replace('/', '_') for s in strategies]
            filename = generator.generate_filename(
                exchange,
                *strategy_names)
            self.results_file = output_dir / filename

        def _load_existing_results(self) -> List[Dict[str, Any]]:
            """Load existing results from file."""
            if not self.results_file.exists():
                logger.info("No existing signals file found")
                return []

            try:
                with open(self.results_file, 'r') as f:
                    logger.info(f"Reading existing signals from {self.results_file}")
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading existing results, starting fresh")
                return []

        def _clean_signal_data(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
            """Clean signal data to remove circular references and non-serializable types."""
            cleaned_data = {}
            for key, value in signal_data.items():
                if isinstance(value, (str, int, float, bool)):
                    cleaned_data[key] = value
                elif isinstance(value, (list, tuple)):
                    cleaned_data[key] = [str(item) if not isinstance(item, (str, int, float, bool, dict)) else item for item in value]
                elif isinstance(value, dict):
                    cleaned_data[key] = self._clean_signal_data(value)
                else:
                    cleaned_data[key] = str(value)
            return cleaned_data

        def save_results(self, new_results: List[SignalData], timestamp: str) -> None:
            """Save new results combined with existing ones."""
            if not new_results:
                return
                
            try:
                # Validate results
                for result in new_results:
                    for field, field_type in self.REQUIRED_FIELDS.items():
                        if field not in result or not isinstance(result[field], field_type):
                            logger.warning(f"Invalid result format: missing or invalid {field}")
                            return
                
                existing_results = self._load_existing_results()
                
                # Clean signal data before combining
                cleaned_new_results = [self._clean_signal_data(result) for result in new_results]
                combined_results = existing_results + cleaned_new_results
                
                # Process in batches to handle large result sets
                for i in range(0, len(combined_results), self.BATCH_SIZE):
                    batch = combined_results[i:i + self.BATCH_SIZE]
                    
                    with open(self.results_file, 'w') as f:
                        logger.info(f"Writing batch of signals to {self.results_file}")
                        json.dump(batch, f, indent=2)
                
                logger.info("Successfully updated combined results file:")
                logger.info(f"- New signals added: {len(new_results)}")
                logger.info(f"- Total signals in file: {len(combined_results)}")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")

        def process_symbol_results(
            self,
            symbol: str,
            exchange: str,
            timestamp: str,
            strategies: List[Strategy],
            data: pd.DataFrame
        ) -> List[Dict[str, Any]]:
            """Process a symbol through all strategies and return results."""
            results = []
            for strategy in strategies:
                try:
                    signals = strategy.analyze(data)
                    if signals:
                        for signal in signals:
                            # Ensure signal is a dict
                            if not isinstance(signal, dict):
                                continue
                            # Use new strength and summary score
                            # Handle strength value which can be string or numeric
                            strength_val = signal.get('Strength', 0.0)
                            strength = 1.0 if isinstance(strength_val, str) and strength_val.lower() == 'strong' else float(strength_val) if not isinstance(strength_val, str) else 0.0
                            
                            summary_score = signal.get('Summary', 'Neutral')
                            
                            # Ensure all required fields are present
                            enhanced_signal = {
                                'Symbol': symbol,
                                'Exchange': exchange,
                                'Timestamp': timestamp,
                                'Strategy': strategy.name,
                                'Type': signal.get('Type', 'UNKNOWN'),
                                'Summary': summary_score,
                                'Price': float(signal.get('Price', 0.0)),
                                'DailyChange': float(signal.get('DailyChange', 0.0)),
                                'WeeklyChange': float(signal.get('WeeklyChange', 0.0)),
                                'Volume': int(signal.get('Volume', 0)),
                                'Signals': signal.get('Signals', {}),
                                'Details': signal.get('Details', {}),
                                'Strength': strength,
                                'Confidence': strength  # Map strength to confidence for compatibility
                            }
                            signal.update(enhanced_signal)
                            results.append(signal)
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed for {symbol}: {e}")
            return results

    class DataHandler:
        """Helper class for data file operations."""
        
        STANDARD_COLUMNS = {
            'close': 'Close',
            'volume': 'Volume',
            'adjClose': 'Adj Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open'
        }

        DATE_COLUMNS = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']
        
        @staticmethod
        def _extract_data(data: Union[Dict, List]) -> Optional[List[Dict]]:
            """Extract data from various JSON formats."""
            if isinstance(data, dict):
                return data.get('data') or data.get('history') or data
            elif isinstance(data, list):
                return data
            return None

        def __init__(self, data_dir: Path):
            self.data_dir = data_dir

        async def load_exchange_data(self, exchange: str) -> pd.DataFrame:
            """Load stock data for a specific exchange."""
            try:
                file_path = self.data_dir / f"{exchange.lower()}_filtered_listings.json"
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                return pd.DataFrame(data).rename(columns={"symbol": "Symbol"})
            except FileNotFoundError:
                logger.error(f"Exchange data file not found: {file_path}")
                return pd.DataFrame()

        def get_historical_data(self, symbol: str, exchange: str) -> Optional[dict]:
            """Get historical data for a symbol from the local data store."""
            try:
                file_path = self.data_dir / exchange / f"{symbol}.json"
                if not file_path.exists():
                    logger.warning(f"No data file found for {symbol}")
                    return None

                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                return self._extract_data(data)
                
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {str(e)}")
                return None

    VALID_EXCHANGES = ['ASX', 'HKEX', 'NASDAQ', 'NYSE']
    
    def __init__(self, config):
        self.config = config
        self.strategies: List[Strategy] = []
        self.market_data = {}  # Store market data for analysis
        self.data_handler: Optional[StockScanner.DataHandler] = None
        
        self._inactive_cache = {}  # Cache for failed validation attempts
        self._cache_ttl = 300  # Cache TTL in seconds (5 minutes)
        # Only initialize these if we have a valid config
        if config is not None:
            self.data_dir = Path(config.data_directory)
            self.output_dir = Path(config.output_directory)
            self.progress_tracker = UpdateProgressTracker(str(self.data_dir))
            self.data_handler = StockScanner.DataHandler(self.data_dir)
            self.yahoo_config = config.providers.get('yahooFinance', {})
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=300  # 5 minutes
            )

    def validate_exchange(self, exchange: str) -> bool:
        """Validate if the exchange is supported."""
        return exchange in self.VALID_EXCHANGES
        
    def _is_cached_inactive(self, symbol: str) -> bool:
        """Check if a symbol is cached as inactive and the cache is still valid."""
        if symbol in self._inactive_cache:
            cache_time, attempts = self._inactive_cache[symbol]
            if time.time() - cache_time < self._cache_ttl:
                return True
            # Clear expired cache entry
            del self._inactive_cache[symbol]
        return False

    def _mark_inactive(self, symbol: str) -> None:
        """Mark a symbol as inactive in the cache."""
        current_time = time.time()
        if symbol in self._inactive_cache:
            _, attempts = self._inactive_cache[symbol]
            # Keep TTL constant but track attempts
            self._inactive_cache[symbol] = (current_time, attempts + 1) 
        else:
            self._inactive_cache[symbol] = (current_time, 1)

    def _should_retry_inactive(self, symbol: str) -> bool:
        """Determine if we should retry checking an inactive symbol."""
        if symbol not in self._inactive_cache:
            return True
        _, attempts = self._inactive_cache[symbol]
        # Retry up to 5 times with increasing delays
        max_attempts = 5
        return attempts < max_attempts

    def register_strategy(self, strategy: Strategy) -> None:
        """Register a new strategy with the scanner."""
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")

    async def scan(self, days: int = 60) -> None:
        """Execute the scanning process using registered strategies."""
        if self.config is None:
            raise ValueError("Scanner not properly initialized with configuration")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.market_data = {}  # Reset market data store
        results = []
        batch_size = 100  # Process symbols in batches

        for exchange_config in self.config.exchanges:
            exchange_name = exchange_config['name']
            logger.info(f"Processing exchange: {exchange_name}")
            
            # Create results manager for this exchange
            results_manager = self.ResultsManager(self.output_dir, exchange_name, self.strategies)

            exchange_data = await self.data_handler.load_exchange_data(exchange_name)
            if exchange_data.empty:
                logger.warning(f"No data found for exchange: {exchange_name}")
                continue

            symbols = exchange_data['Symbol'].tolist()
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_results = []

                for symbol in batch:
                    try:
                        historical_data = self.data_handler.get_historical_data(symbol, exchange_name)
                        if historical_data is None:
                            continue

                        cleaned_data = clean_historical_data({'data': historical_data})
                        if cleaned_data is None:
                            logger.warning(f"Could not clean data for {symbol}")
                            continue

                        # Store market data for analysis
                        self.market_data[symbol] = cleaned_data

                        symbol_results = results_manager.process_symbol_results(
                            symbol, exchange_name, timestamp, self.strategies, cleaned_data
                        )
                        batch_results.extend(symbol_results)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")

                results.extend(batch_results)
                results_manager.save_results(batch_results, timestamp)

        logger.info(f"Scan completed. Found {len(results)} total signals.")
        logger.info("Generating trading analysis report...")
        # Generate trading analysis report
        self._generate_analysis_report(results)
        
    def _generate_analysis_report(self, signals: List[Dict[str, Any]]) -> None:
        """Generate and save comprehensive trading analysis report."""
        try:
            # Pre-process signals to ensure all fields are properly formatted
            processed_signals = []
            for signal in signals:
                # Convert Timestamp objects to ISO format strings
                if isinstance(signal.get('Timestamp'), pd.Timestamp):
                    signal['Timestamp'] = signal['Timestamp'].isoformat()
                
                # Ensure Strength is a float
                strength_val = signal.get('Strength', 0.0)
                signal['Strength'] = (1.0 if isinstance(strength_val, str) and strength_val.lower() == 'strong'
                                    else float(strength_val) if not isinstance(strength_val, str)
                                    else 0.0)
                
                # Ensure other numeric fields are properly formatted
                signal['Price'] = float(signal.get('Price', 0.0))
                signal['Volume'] = int(signal.get('Volume', 0))
                signal['DailyChange'] = float(signal.get('DailyChange', 0.0))
                signal['WeeklyChange'] = float(signal.get('WeeklyChange', 0.0))
                
                processed_signals.append(signal)
            analyzer = TradingAnalysis(processed_signals, self.market_data)
            report = analyzer.generate_full_report()
            
            # Convert the report to be JSON serializable
            def json_serialize(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                return str(obj)
            
            # Save report with microsecond precision removed from timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"trading_analysis_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=json_serialize)
            
            logger.info(f"Trading analysis report saved to {report_path}")
            return report
        except Exception as e:
            logger.error(f"Error generating trading analysis report: {str(e)}")
            return None

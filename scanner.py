from typing import List, Optional, Dict, Any, Tuple, Iterator, TypedDict, Union
import pandas as pd
import numpy as np
import logging
import time
import pytz
from datetime import datetime, timedelta, time as dt_time
import yfinance as yf
from utils.yfinance_utils import get_cached_session
from pathlib import Path
import json
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from strategies.base import Strategy
from utils.async_utils import YahooFinanceClient, CircuitBreaker, UpdateProgressTracker
from utils.data_cleaner import clean_historical_data

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
            """
            Get the most recent trading day for a given market timezone.
            
            Args:
                tz_name: Timezone name (e.g., 'US/Eastern', 'Australia/Sydney')
                market_open: Market opening time
                market_close: Market closing time
                validation_symbol: Symbol to validate trading day against (e.g., 'SPY', 'STW.AX')
                check_data: Whether to validate against market data
            """
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
                file_path = self.data_dir / f"{exchange.lower()}_valid_listings.json"
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                return pd.DataFrame(data).rename(columns={"symbol": "Symbol"})
            except FileNotFoundError:
                logger.error(f"Exchange data file not found: {file_path}")
                return pd.DataFrame()
            
        async def load_stock_data(self, file_path: Path) -> Optional[pd.DataFrame]:
            """Load and parse stock data from a JSON file."""
            if not file_path.exists():
                return None

            try:
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                
                data_list = self._extract_data(data)
                if not data_list:
                    logger.error(f"Expected list of data points for {file_path}, got {type(data)}")
                    return None
                
                df = pd.DataFrame(data_list)
                if 'date' not in df.columns:
                    for col in self.DATE_COLUMNS:
                        if col in df.columns:
                            df['date'] = df[col]
                            break
                    else:
                        logger.error(f"No date column found in data for {file_path}")
                        return None
                
                df = self._process_dataframe(df)
                df = df.reindex(date_range).ffill().dropna()
                df = df[~df.index.duplicated(keep='last')]

                return df.sort_index()

            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {str(e)}")
                return None
                
        def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
            """Process a DataFrame to standardize format and handle missing values."""
            # Convert and set date index
            df['Date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df.set_index('Date', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            
            # Standardize column names
            df.rename(columns=self.STANDARD_COLUMNS, inplace=True)
            
            # Create continuous date range
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            return df

        async def save_stock_data(self, file_path: Path, data: Dict[str, Any]) -> None:
            """Save stock data to a JSON file."""
            try:
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
            except Exception as e:
                logger.error(f"Error saving data to {file_path}: {str(e)}")
                raise

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

    class ResultsManager:
        """Manages storage and retrieval of scan results."""
        RESULTS_FILENAME = "combined_macd_signals.json"
        BATCH_SIZE = 1000  # Number of results to process at once
        
        REQUIRED_FIELDS = {
            'Symbol': str,
            'Exchange': str,
            'Timestamp': str,
            'Strategy': str
        }
        
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.results_file = output_dir / self.RESULTS_FILENAME

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
                combined_results = existing_results + new_results
                
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
                            signal.update({
                                'Symbol': symbol,
                                'Exchange': exchange,
                                'Timestamp': timestamp,
                                'Strategy': strategy.name
                            
                            })
                            results.append(signal)
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed for {symbol}: {e}")
            return results

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

    async def _load_exchange_data(self, exchange: str) -> pd.DataFrame:
        """Load stock data for a specific exchange."""
        try:
            file_path = self.data_dir / f"{exchange.lower()}_valid_listings.json"
            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
            return pd.DataFrame(data).rename(columns={"symbol": "Symbol"})
        except FileNotFoundError:
            logger.error(f"Exchange data file not found: {file_path}")
            return pd.DataFrame()

    async def _load_stock_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse stock data from a JSON file."""
        try:
            # For new files, return None without logging an error
            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
            
            # Handle different data formats
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'history' in data:
                    data = data['history']
            
            # Ensure data is in the correct format
            if not isinstance(data, list):
                logger.error(f"Expected list of data points for {file_path}, got {type(data)}")
                return None
            
            df = pd.DataFrame(data)
            
            # Verify and standardize date column
            if 'date' not in df.columns:
                # Try alternate date column names
                date_columns = ['Date', 'DATE', 'timestamp', 'Timestamp']
                for col in date_columns:
                    if col in df.columns:
                        df['date'] = df[col]
                        break
                else:
                    logger.error(f"No date column found in data for {file_path}")
                    return None
            
            # Convert date and standardize column names
            dates = pd.to_datetime(df['date'])
            df['Date'] = dates.dt.tz_localize(None)
            df.set_index('Date', inplace=True)
            
            # Make index unique by keeping last value
            df = df[~df.index.duplicated(keep='last')]
            
            # Standardize column names
            df.rename(columns={
                'close': 'Close',
                'volume': 'Volume',
                'adjClose': 'Adj Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open'
            }, inplace=True)
            
            # Sort index and handle missing values
            df = df.sort_index()
            
            # Create continuous date range and forward fill missing values
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(date_range)
            df = df.ffill()  # Forward fill missing values
            
            # Drop any remaining NaN and duplicate indices
            df = df.dropna()
            df = df[~df.index.duplicated(keep='last')]
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None

    def _get_last_trading_day(self, check_data: bool = True) -> datetime:
        """Get the most recent trading day for US markets."""
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        return self.MarketTimes.get_trading_day(
            'US/Eastern',
            market_open,
            market_close,
            'SPY',
            check_data
        )

    def _get_asx_trading_day(self, check_data: bool = True) -> datetime: 
        """Get the most recent trading day for the ASX market."""
        return self.MarketTimes.get_trading_day(
            'Australia/Sydney',
            dt_time(10, 0),
            dt_time(16, 0),
            'STW.AX',
            check_data
        )

    async def _process_symbol(
        self,
        client: YahooFinanceClient,
        symbol: str,
        exchange_name: str,
        exchange_dir: Path,
        end_date: datetime
    ) -> bool:
        """Process a single symbol update for data collection."""
        # Check cache first
        if self._is_cached_inactive(symbol) and not self._should_retry_inactive(symbol):
            logger.info(f"Skipping {symbol} (cached inactive)")
            return True
            
        try:
            file_path = exchange_dir / f"{symbol}.json"
            existing_df = await self.data_handler.load_stock_data(file_path)
            start_date = None if existing_df is None else existing_df.index[-1] + timedelta(days=1)
            
            if start_date:
                start_date = pd.Timestamp(start_date).tz_localize(pytz.UTC)
            end_date_utc = end_date.astimezone(pytz.UTC)
            
            if start_date and start_date >= pd.Timestamp(end_date_utc.date(), tz=pytz.UTC):
                return True  # Data is already up to date

            # Handle exchange-specific symbol formatting
            yahoo_symbol = symbol
            if exchange_name == 'ASX' and not symbol.endswith('.AX'):
                yahoo_symbol = f"{symbol}.AX"
            elif exchange_name == 'HKEX' and not symbol.endswith('.HK'):
                yahoo_symbol = f"{symbol}.HK"
                
            # Get basic info first
            logger.info(f"Fetching info for {yahoo_symbol}")
            info = await self.circuit_breaker(client.get_ticker_info, yahoo_symbol)

            # Use YahooFinanceClient's validation logic
            if not client._is_valid_stock_data(info):
                logger.info(f"Symbol {symbol} validation failed - reason: not an active equity/ETF")
                self._mark_inactive(symbol)
                # Log more details for debugging
                if info and 'price' in info:
                    price_data = info['price']
                    logger.debug(f"Validation details for {symbol}: marketState={price_data.get('marketState')}, quoteType={price_data.get('quoteType')}, tradingStatus={price_data.get('tradingStatus')}")
                return True

            # Get historical data
            query_end_date = end_date + timedelta(days=1)
            end_date_str = query_end_date.astimezone(pytz.UTC).strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
            
            logger.debug(f"Fetching history for {yahoo_symbol}")
            logger.debug(f"Start date: {start_date_str}, End date: {end_date_str}")
            
            history_data = await self.circuit_breaker(
                client.get_history,
                yahoo_symbol,
                start_date_str,
                end_date_str
            )

            if not history_data or 'chart' not in history_data or not history_data['chart'].get('result'):
                logger.info(f"No new data available for {symbol}")
                return True

            # Process the data
            result = history_data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            adjclose = result['indicators']['adjclose'][0] if 'adjclose' in result['indicators'] else None

            data_list = []
            for i, timestamp in enumerate(timestamps):
                if pd.isna(quote['close'][i]):
                    continue
                    
                data_point = {
                    'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                    'open': quote['open'][i],
                    'high': quote['high'][i],
                    'low': quote['low'][i],
                    'close': quote['close'][i],
                    'volume': quote['volume'][i],
                    'adjClose': adjclose['adjclose'][i] if adjclose else quote['close'][i]
                }
                data_list.append(data_point)

            if existing_df is not None:
                # Merge with existing data
                existing_data = json.loads(existing_df.to_json(orient='records', date_format='iso'))
                data_list = existing_data + data_list

            await self.data_handler.save_stock_data(file_path, {'data': data_list})
            logger.info(f"Updated data for {symbol} ({len(data_list)} records)")
            return True

        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {str(e)}")
            return False

    async def update_stock_data(self) -> None:
        """Update historical data for all monitored securities."""
        if self.config is None:
            raise ValueError("Scanner not properly initialized with configuration")

        rate_limit = self.yahoo_config.get('rateLimit', 2000)
        batch_size = min(self.yahoo_config.get('batchSize', 3), 3)  # Cap batch size at 3
        
        await self.progress_tracker.load_progress()
        
        async with YahooFinanceClient(rate_limit, batch_size) as client:
            tasks = []
            logger.debug(f"Exchanges configuration: {self.config.exchanges}")
            
            for exchange_config in self.config.exchanges:
                exchange_name = exchange_config['name']
                logger.info(f"Updating data for exchange: {exchange_name}")
                
                # Load exchange data
                exchange_data = await self.data_handler.load_exchange_data(exchange_name)
                if exchange_data.empty:
                    logger.warning(f"No symbols found for exchange: {exchange_name}")
                    continue
                
                # Create exchange directory if it doesn't exist
                exchange_dir = self.data_dir / exchange_name
                exchange_dir.mkdir(exist_ok=True)
                
                # Get the appropriate end date for the exchange
                if exchange_name == 'ASX':
                    end_date = self._get_asx_trading_day(check_data=True)
                else:
                    end_date = self._get_last_trading_day(check_data=False)
                
                # Create tasks for each symbol
                exchange_tasks = []
                total_symbols = len(exchange_data['Symbol'])
                completed_symbols = 0
                
                for symbol in exchange_data['Symbol']:
                    task = asyncio.create_task(self._process_symbol(
                        client,
                        symbol,
                        exchange_name,
                        exchange_dir,
                        end_date
                    ))
                    exchange_tasks.append(task)
                    
                    if len(exchange_tasks) >= batch_size:
                        # Wait for the batch to complete
                        await asyncio.sleep(2)  # Add delay between batches
                        results = await asyncio.gather(*exchange_tasks, return_exceptions=True)
                        completed_symbols += len(exchange_tasks)
                        
                        # Update progress
                        await self.progress_tracker.update_exchange_progress(
                            exchange_name,
                            total_symbols,
                            completed_symbols
                        )
                        
                        # Record any failures
                        for symbol, result in zip(exchange_data['Symbol'].iloc[completed_symbols-len(exchange_tasks):completed_symbols], results):
                            if isinstance(result, Exception):
                                logger.error(f"Error processing {symbol}: {str(result)}")
                                if "Too Many Requests" in str(result):
                                    await asyncio.sleep(30)  # Additional backoff for rate limits
                                await self.progress_tracker.add_failed_symbol(exchange_name, symbol)
                            elif result is False:
                                await self.progress_tracker.add_failed_symbol(exchange_name, symbol)
                        
                        exchange_tasks = []
                
                # Process any remaining symbols
                if exchange_tasks:
                    results = await asyncio.gather(*exchange_tasks, return_exceptions=True)
                    completed_symbols += len(exchange_tasks)
                    
                    await self.progress_tracker.update_exchange_progress(
                        exchange_name,
                        total_symbols,
                        completed_symbols
                    )
                    
                    for symbol, result in zip(exchange_data['Symbol'].iloc[completed_symbols-len(exchange_tasks):completed_symbols], results):
                        if isinstance(result, Exception) or result is False:
                            await self.progress_tracker.add_failed_symbol(exchange_name, symbol)
                
                logger.info(f"Completed updating data for exchange: {exchange_name}")

    async def scan(self, days: int = 60) -> None:
        """Execute the scanning process using registered strategies."""
        if self.config is None:
            raise ValueError("Scanner not properly initialized with configuration")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        results_manager = self.ResultsManager(self.output_dir)
        results = []
        batch_size = 100  # Process symbols in batches

        for exchange_config in self.config.exchanges:
            exchange_name = exchange_config['name']
            logger.info(f"Processing exchange: {exchange_name}")

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

                        symbol_results = results_manager.process_symbol_results(
                            symbol, exchange_name, timestamp, self.strategies, cleaned_data
                        )
                        batch_results.extend(symbol_results)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")

                results.extend(batch_results)
                results_manager.save_results(batch_results, timestamp)

        logger.info(f"Scan completed. Found {len(results)} total signals.")

from typing import List, Optional
import pandas as pd
import logging
import pytz
from datetime import datetime, timedelta, time as dt_time
import yfinance as yf
from pathlib import Path
import json

from strategies.base import Strategy

logger = logging.getLogger(__name__)

class StockScanner:
    VALID_EXCHANGES = ['ASX', 'HKEX', 'NASDAQ', 'NYSE']
    
    def __init__(self, config):
        self.config = config
        self.strategies: List[Strategy] = []
        self.data_dir = None
        self.output_dir = None
        if config is not None:
            self.data_dir = Path(config.data_directory)
            self.output_dir = Path(config.output_directory)

    def validate_exchange(self, exchange: str) -> bool:
        """Validate if the exchange is supported."""
        return exchange in self.VALID_EXCHANGES

    def register_strategy(self, strategy: Strategy) -> None:
        """Register a new strategy with the scanner."""
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")

    def _load_exchange_data(self, exchange: str) -> pd.DataFrame:
        """Load stock data for a specific exchange."""
        try:
            file_path = self.data_dir / f"{exchange.lower()}_valid_listings.json"
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data).rename(columns={"symbol": "Symbol"})
        except FileNotFoundError:
            logger.error(f"Exchange data file not found: {file_path}")
            return pd.DataFrame()

    def _load_stock_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and parse stock data from a JSON file."""
        try:
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract the data array if it's in the metadata format
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            elif isinstance(data, dict) and 'history' in data:
                data = data['history']
            
            # Ensure data is in the correct format
            if not isinstance(data, list):
                logger.error(f"Expected list of data points for {file_path}, got {type(data)}")
                return None
            
            df = pd.DataFrame(data)
            
            # Verify date column exists
            if 'date' not in df.columns:
                logger.error(f"No date column found in data for {file_path}")
                return None
            
            # Convert date and standardize column names
            df['Date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Ensure dates are timezone naive
            df.set_index('Date', inplace=True)
            
            # Standardize column names
            df.rename(columns={
                'close': 'Close',
                'volume': 'Volume',
                'adjClose': 'Adj Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open'
            }, inplace=True)
            
            return df.sort_index()
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None

    def _get_last_trading_day(self, check_data: bool = True) -> datetime:
        """Get the most recent trading day for US markets."""
        # Get current time in US/Eastern timezone (for US markets)
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        current_day = current_time.date()
        proposed_trading_day = None
        
        # Check if it's weekend
        if current_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            days_to_subtract = current_time.weekday() - 4  # Go back to Friday
            last_trading = current_time - timedelta(days=days_to_subtract)
            proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
            
        # Check if it's before market open (9:30 AM EST) or after market close (4:00 PM EST)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_est_time = current_time.time()
        
        if proposed_trading_day is None and (current_est_time < market_open or current_est_time >= market_close):
            last_trading = current_time - timedelta(days=1)
            proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if proposed_trading_day is None:
            proposed_trading_day = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
        # If we don't need to verify against actual data, return the proposed day
        if not check_data:
            return proposed_trading_day
            
        # Verify against actual market data
        try:
            # Use a major stock like SPY to check if market was open
            spy = yf.Ticker("SPY")
            # Get data for the last 5 days to find the most recent trading day
            end_date = proposed_trading_day.astimezone(pytz.UTC).strftime('%Y-%m-%d')
            df = spy.history(period="5d", end=end_date)
            
            if df.empty:
                logger.warning("No recent market data available to verify trading day")
                return proposed_trading_day
                
            # Get the last actual trading day from the data
            last_actual_trading = df.index[-1].to_pydatetime().astimezone(est)
            return last_actual_trading.replace(hour=16, minute=0, second=0, microsecond=0)
            
        except Exception as e:
            logger.error(f"Error verifying trading day against market data: {str(e)}")
            return proposed_trading_day

    def _get_most_recent_trading_day_with_data(self, exchange_name: str) -> datetime:
        """Get the most recent trading day with available data for the exchange."""
        exchange_dir = self.data_dir / exchange_name
        if not exchange_dir.exists():
            return None
        
        # Get all JSON files in the exchange directory
        data_files = list(exchange_dir.glob("*.json"))
        if not data_files:
            return None
            
        return max(f.stat().st_mtime for f in data_files)

    def _get_asx_trading_day(self, check_data: bool = True) -> datetime:
        """Get the most recent trading day for ASX market."""
        # Get current time in Australia/Sydney timezone
        sydney_tz = pytz.timezone('Australia/Sydney')
        current_time = datetime.now(sydney_tz)
        current_day = current_time.date()
        proposed_trading_day = None
        
        # Check if it's weekend
        if current_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            days_to_subtract = current_time.weekday() - 4  # Go back to Friday
            last_trading = current_time - timedelta(days=days_to_subtract)
            proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
            
        # Check if it's before market open (10:00 AM AEST) or after market close (4:00 PM AEST)
        market_open = dt_time(10, 0)
        market_close = dt_time(16, 0)
        current_syd_time = current_time.time()
        
        if proposed_trading_day is None and (current_syd_time < market_open or current_syd_time >= market_close):
            last_trading = current_time - timedelta(days=1)
            proposed_trading_day = last_trading.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if proposed_trading_day is None:
            proposed_trading_day = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
        # If we don't need to verify against actual data, return the proposed day
        if not check_data:
            return proposed_trading_day
            
        # Verify against actual data by checking the most recent file modification time
        most_recent_data_day = self._get_most_recent_trading_day_with_data('ASX')
        if most_recent_data_day:
            most_recent_data_day = datetime.fromtimestamp(most_recent_data_day)
            # Convert to Sydney timezone
            most_recent_data_day = most_recent_data_day.astimezone(sydney_tz)
            if most_recent_data_day > proposed_trading_day:
                return most_recent_data_day
                
        return proposed_trading_day

    def _get_historical_data(self, symbol: str, exchange_name: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Load historical data for a symbol from local JSON file."""
        file_path = self.data_dir / exchange_name / f"{symbol}.json"
        df = self._load_stock_data(file_path)
        return df.tail(days) if df is not None else None

    def update_stock_data(self) -> None:
        """Update historical data for all monitored securities."""
        if self.config is None:
            raise ValueError("Scanner not properly initialized with configuration")

        yahoo_config = self.config.providers.get('yahooFinance', {})
        rate_limit = yahoo_config.get('rateLimit', 2000) / 1000  # Convert to seconds
        proxy_config = yahoo_config.get('proxy', None)
        
        # Add exponential backoff for retries
        base_delay = rate_limit
        max_retries = 3

        for exchange_config in self.config.exchanges:
            exchange_name = exchange_config['name']
            logger.info(f"Updating data for exchange: {exchange_name}")

            # Load exchange data
            exchange_data = self._load_exchange_data(exchange_name)
            if exchange_data.empty:
                logger.warning(f"No symbols found for exchange: {exchange_name}")
                continue

            # Create exchange directory if it doesn't exist
            exchange_dir = self.data_dir / exchange_name
            exchange_dir.mkdir(exist_ok=True)

            # Process each symbol
            for symbol in exchange_data['Symbol']:
                file_path = exchange_dir / f"{symbol}.json"
                
                # Check if the ticker is still valid/listed before attempting to fetch data
                if symbol not in exchange_data['Symbol'].values:
                    logger.info(f"Skipping {symbol} as it appears to be delisted")
                    continue

                # Check if we should update this symbol
                # Get existing data
                existing_df = self._load_stock_data(file_path)
                start_date = None if existing_df is None else existing_df.index[-1] + timedelta(days=1)
                if exchange_name == 'ASX':
                    end_date = self._get_asx_trading_day(check_data=True)
                else:
                    end_date = self._get_last_trading_day(check_data=False)
                
                # Convert dates to UTC for comparison
                if start_date:
                    start_date = pd.Timestamp(start_date).tz_localize(pytz.UTC)
                end_date_utc = end_date.astimezone(pytz.UTC)
                
                if start_date and start_date >= pd.Timestamp(end_date_utc.date(), tz=pytz.UTC):
                    continue  # Skip if data is already up to date

                success = False
                attempt = 0
                while not success and attempt < max_retries:
                    try:
                        yahoo_symbol = f"{symbol}.AX" if exchange_name == 'ASX' else symbol
                        
                        # First verify if the symbol is still active by attempting to get basic info
                        stock = yf.Ticker(yahoo_symbol)
                        info = stock.get_info()
                        
                        if not info or 'regularMarketPrice' not in info or info.get('regularMarketPrice') is None:
                            logger.info(f"Skipping {symbol} as it appears to be inactive")
                            success = True
                            break
                            
                        # Proceed with history fetching
                        query_end_date = end_date + timedelta(days=1)
                        end_date_str = query_end_date.astimezone(pytz.UTC).strftime('%Y-%m-%d')
                        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else "max"
                        new_df = stock.history(start=start_date_str, end=end_date_str)

                        if new_df.empty:
                            logger.info(f"No new data available for {symbol}")
                            success = True
                            break

                        current_time = datetime.now(pytz.UTC)
                        new_df = new_df[new_df.index <= current_time]
                        if new_df.empty:
                            success = True
                            break

                        new_df.index = pd.to_datetime(new_df.index).tz_localize(None)

                        if existing_df is not None:
                            df = pd.concat([existing_df, new_df])
                            df = df[~df.index.duplicated(keep='last')]
                        else:
                            df = new_df

                        data_list = []
                        for idx, row in df.iterrows():
                            data_point = {
                                'date': idx.strftime('%Y-%m-%d'),
                                'open': row['Open'],
                                'high': row['High'],
                                'low': row['Low'],
                                'close': row['Close'],
                                'volume': row['Volume'],
                                'adjClose': row['Adj Close'] if 'Adj Close' in row else row['Close']
                            }
                            data_list.append(data_point)

                        with open(file_path, 'w') as f:
                            json.dump({'data': data_list}, f, indent=2)

                        logger.info(f"Updated data for {symbol} ({len(data_list)} records)")
                        import time; time.sleep(rate_limit)
                        success = True
                    except Exception as e:
                        if "Failed to establish a new connection" in str(e):
                            attempt += 1
                            if attempt < max_retries:
                                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(f"Network connectivity issue for {symbol}: {str(e)}. Retrying in {retry_delay:.1f}s...")
                                import time; time.sleep(retry_delay)
                            else:
                                logger.error(f"Persistent network connectivity issue for {symbol}: {str(e)}. Skipping ticker.")
                                break
                        else:
                            logger.error(f"Error updating data for {symbol}: {str(e)}")
                            break
                if not success:
                    continue

            logger.info(f"Completed updating data for exchange: {exchange_name}")

    def scan(self, days: int = 60) -> None:
        """Execute the scanning process using registered strategies."""
        if self.config is None:
            raise ValueError("Scanner not properly initialized with configuration")
            
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        results = []

        for exchange_config in self.config.exchanges:
            exchange_name = exchange_config['name']
            logger.info(f"Processing exchange: {exchange_name}")

            # Load exchange data
            exchange_data = self._load_exchange_data(exchange_name)
            if exchange_data.empty:
                logger.warning(f"No data found for exchange: {exchange_name}")
                continue

            # Process each symbol
            for symbol in exchange_data['Symbol']:
                historical_data = self._get_historical_data(symbol, exchange_name, days)
                if historical_data is None:
                    continue

                # Apply each strategy
                for strategy in self.strategies:
                    try:
                        signals = strategy.analyze(historical_data)
                        if signals:
                            for signal in signals:
                                signal['Symbol'] = symbol
                                signal['Exchange'] = exchange_name
                                signal['Timestamp'] = timestamp
                                results.append(signal)
                    except Exception as e:
                        logger.error(f"Strategy {strategy.name} failed for {symbol}: {e}")

        # Save results
        if results:
            self._save_results(results, timestamp)
        
        logger.info(f"Scan completed. Found {len(results)} signals.")

    def _save_results(self, results: List[dict], timestamp: str) -> None:
        """Save scanning results to CSV and JSON files."""
        if not results:
            return

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(results)
        
        # Combine symbol and exchange, then drop separate exchange column
        df['Symbol'] = df.apply(lambda x: f"{x['Symbol']}:{x['Exchange']}", axis=1)
        df.drop('Exchange', axis=1, inplace=True)
        
        # Save as CSV
        csv_path = self.output_dir / f"macd_signals_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")

        # Save as JSON
        json_path = self.output_dir / f"macd_signals_{timestamp}.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Results saved to JSON: {json_path}")

import json
import os
import pandas as pd
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Add file handler
    file_handler = logging.FileHandler('logs/exchange_data.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ExchangeDataLoader:
    """Loads and processes market data from different exchanges."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            base_dir: Base directory containing exchange data folders
        """
        self.base_dir = base_dir
        self.exchanges = ['ASX', 'NYSE', 'NASDAQ', 'HKEX']
        
    def load_stock_data(self, exchange: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific stock.
        
        Args:
            exchange: Exchange name (ASX, NYSE, NASDAQ, HKEX)
            symbol: Stock symbol
            
        Returns:
            DataFrame with historical data or None if data is invalid/missing
        """
        try:
            # First, try to load from the exchange-specific directory (JSON format)
            file_path = os.path.join(self.base_dir, exchange.lower(), f"{symbol}.json")
            
            # Check if file exists
            if os.path.exists(file_path):
                # Load JSON data
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        logger.debug(f"Successfully loaded JSON data from {file_path}")
                        
                        # Convert to DataFrame based on the data structure
                        df = self._convert_to_dataframe(data, file_path)
                        if df is not None:
                            # Process and validate DataFrame
                            return self._process_dataframe(df)
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding JSON from {file_path}")
            else:
                logger.warning(f"Data file not found: {file_path}")
            
            # If we didn't successfully load from the exchange directory,
            # try to load from the market_data directory (CSV format)
            market_data_paths = [
                os.path.join(self.base_dir, "market_data", "1d", "raw", f"{symbol}.csv"),
                os.path.join(self.base_dir, "market", "1d", "raw", f"{symbol}.csv"),
                os.path.join("data", "market_data", "1d", "raw", f"{symbol}.csv"),
                os.path.join("data", "market", "1d", "raw", f"{symbol}.csv")
            ]
            
            for csv_path in market_data_paths:
                if os.path.exists(csv_path):
                    logger.info(f"Loading CSV data from {csv_path}")
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # Process the CSV format data
                        if 'date' in df.columns:
                            # Handle date column
                            df['date'] = pd.to_datetime(df['date'], utc=True)
                            
                            # Make sure all column names are lowercase
                            df.columns = [col.lower() for col in df.columns]
                            
                            # Make sure we have all required columns
                            required_cols = ['open', 'high', 'low', 'close', 'volume']
                            if all(col in df.columns for col in required_cols):
                                # Sort by date
                                df = df.sort_values('date')
                                
                                # Add some debugging info
                                logger.info(f"Loaded {len(df)} rows of data for {symbol}")
                                logger.info(f"Columns: {list(df.columns)}")
                                
                                return df
                            else:
                                logger.warning(f"Missing required columns in {csv_path}")
                    except Exception as e:
                        logger.warning(f"Error loading CSV from {csv_path}: {str(e)}")
            
            logger.warning(f"No data found for {symbol} in any location")
            return None
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def _convert_to_dataframe(self, data: Any, file_path: str) -> Optional[pd.DataFrame]:
        """
        Convert JSON data to DataFrame based on its structure.
        Handles multiple common JSON structures for stock data.
        
        Args:
            data: Loaded JSON data
            file_path: Path to the data file (for error reporting)
            
        Returns:
            DataFrame or None if conversion failed
        """
        try:
            # NYSE Type: {"history": [{date: "...", open: x, high: y, ...}, ...]}
            if isinstance(data, dict) and 'history' in data and isinstance(data['history'], list):
                history_data = data['history']
                if len(history_data) > 0 and all(isinstance(item, dict) for item in history_data):
                    logger.debug(f"Using NYSE history array structure for {file_path}")
                    return pd.DataFrame(history_data)
                
            # Type 1: {code: "...", history: [{date: "...", open: "...", ...}, ...]}
            if isinstance(data, dict) and 'code' in data and 'history' in data and isinstance(data['history'], list):
                logger.debug(f"Using Type 1 data structure for {file_path}")
                return pd.DataFrame(data['history'])
                
            # Type 2: [{date: "...", open: "...", ...}, ...]
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                logger.debug(f"Using Type 2 data structure for {file_path}")
                return pd.DataFrame(data)
                
            # Type 3: {"2023-01-01": {"open": 10, "high": 11, ...}, ...}
            elif isinstance(data, dict) and len(data) > 0:
                # Check if values are dictionaries with OHLCV data
                first_key = next(iter(data))
                first_value = data[first_key]
                
                if isinstance(first_value, dict) and any(k.lower() in ['open', 'high', 'low', 'close'] for k in first_value):
                    logger.debug(f"Using Type 3 data structure for {file_path}")
                    # Convert to list of dictionaries with date field
                    records = []
                    for date_str, values in data.items():
                        record = {'date': date_str}
                        record.update(values)
                        records.append(record)
                    return pd.DataFrame(records)
            
            # Type 4: YFinance format - {'Open': [...], 'High': [...], ...}
            elif isinstance(data, dict) and all(k in data for k in ['Open', 'High', 'Low', 'Close', 'Volume']) and all(isinstance(data[k], list) for k in ['Open', 'High', 'Low', 'Close', 'Volume']):
                logger.debug(f"Using Type 4 data structure for {file_path}")
                # Check if 'Date' or 'Timestamp' exists
                date_key = 'Date' if 'Date' in data else 'Timestamp' if 'Timestamp' in data else None
                
                if date_key:
                    df = pd.DataFrame(data)
                    return df
                else:
                    logger.error(f"Type 4 structure but no date column in {file_path}")
                    return None
            
            # Type 5: Nested structure with 'chart' or 'data' field
            elif isinstance(data, dict) and ('chart' in data or 'data' in data):
                key = 'chart' if 'chart' in data else 'data'
                nested_data = data[key]
                
                # Recursively try to convert nested structure
                logger.debug(f"Trying to convert nested structure with key '{key}' in {file_path}")
                return self._convert_to_dataframe(nested_data, file_path)
            
            logger.error(f"Unrecognized data structure in {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            return None
            
    def _process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process and validate DataFrame."""
        try:
            # Handle different date column names
            date_columns = ['date', 'Date', 'timestamp', 'Timestamp', 'datetime', 'Datetime']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                if df.index.name in date_columns or isinstance(df.index, pd.DatetimeIndex):
                    # Date is already the index
                    pass
                else:
                    logger.error(f"No date column found in DataFrame")
                    return None
            else:
                # Convert date strings to datetime
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Set date as index
                df = df.set_index(date_col)
            
            # Identify price and volume columns if not standard names
            column_mapping = self._identify_columns(df)
            
            # Rename columns to match expected format
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist and are numeric
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    # Check for lowercase versions
                    lowercase_col = col.lower()
                    if lowercase_col in df.columns:
                        df = df.rename(columns={lowercase_col: col})
                    else:
                        logger.error(f"Missing required column: {col}")
                        return None
                        
                # Ensure columns are numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Validate numeric conversion
            nan_counts = df[required_columns].isna().sum()
            if nan_counts.any():
                logger.warning(f"NaN values after numeric conversion: {nan_counts}")
                
            # Drop rows with NaN values
            original_len = len(df)
            df = df.dropna(subset=required_columns)
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN values")
            
            # Sort by date
            df = df.sort_index()
            logger.debug(f"Final DataFrame shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}", exc_info=True)
            return None
            
    def _identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify which columns correspond to Open, High, Low, Close, Volume.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Dictionary mapping source column names to standard names
        """
        column_mapping = {}
        columns_lower = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Common variations of column names
        open_variations = ['open', 'opens', 'opening', 'openprice', 'open price']
        high_variations = ['high', 'highs', 'highest', 'highprice', 'high price']
        low_variations = ['low', 'lows', 'lowest', 'lowprice', 'low price']
        close_variations = ['close', 'closes', 'closing', 'closeprice', 'close price', 'adj close', 'adjusted close', 'adjclose']
        volume_variations = ['volume', 'vol', 'volumes']
        
        # Find matching column names
        for i, col_lower in enumerate(columns_lower):
            if col_lower in open_variations:
                column_mapping[df.columns[i]] = 'Open'
            elif col_lower in high_variations:
                column_mapping[df.columns[i]] = 'High'
            elif col_lower in low_variations:
                column_mapping[df.columns[i]] = 'Low'
            elif col_lower in close_variations:
                column_mapping[df.columns[i]] = 'Close'
            elif col_lower in volume_variations:
                column_mapping[df.columns[i]] = 'Volume'
        
        return column_mapping
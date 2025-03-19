"""
Data provider for historical market data.
Handles data loading, preprocessing, and caching for backtesting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import json
import glob
from influxdb_client import InfluxDBClient
import sys
import traceback
import pytz

# Add the src directory to the path to import from other modules
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.market_data.core.influxdb_cache import load_config, DEFAULT_INFLUXDB_URL, DEFAULT_INFLUXDB_TOKEN

logger = logging.getLogger(__name__)

__all__ = ['DataProvider', 'load_storage_config']

def load_storage_config():
    """Load storage configuration from storage_config.json file."""
    config_path = Path("config") / "storage_config.json"
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded storage configuration from {config_path}")
            return config
        else:
            logger.warning(f"Config file {config_path} not found")
            return None
    except Exception as e:
        logger.error(f"Error loading storage config: {str(e)}")
        return None

class DataProvider:
    """Data provider for backtesting."""
    
    def __init__(self, base_path: str = "data", use_influxdb: bool = True):
        """Initialize data provider.
        
        Args:
            base_path: Base path for data files
            use_influxdb: Whether to use InfluxDB as data source
        """
        self.base_path = Path(base_path)
        self._cache = {}
        
        # Initialize InfluxDB client if enabled
        self._influxdb_client = None
        self._query_api = None
        self._bucket_name = None
        
        if use_influxdb:
            try:
                # First try loading from storage_config.json
                config = load_storage_config()
                if config and 'influxdb' in config:
                    influxdb_config = config['influxdb']
                    url = influxdb_config.get('url', DEFAULT_INFLUXDB_URL)
                    token = influxdb_config.get('token', DEFAULT_INFLUXDB_TOKEN)
                    org = influxdb_config.get('org', 'my-org')
                    self._bucket_name = influxdb_config.get('bucket', 'stocks')
                else:
                    # Use defaults if no config file
                    url = DEFAULT_INFLUXDB_URL
                    token = DEFAULT_INFLUXDB_TOKEN
                    org = 'my-org'
                    self._bucket_name = 'stocks'
                
                self._influxdb_client = InfluxDBClient(
                    url=url,
                    token=token,
                    org=org,
                    timeout=30_000
                )
                self._query_api = self._influxdb_client.query_api()
                logger.info(f"Successfully initialized InfluxDB client with bucket: {self._bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB client: {str(e)}")
                logger.error(traceback.format_exc())
                self._influxdb_client = None

    def _get_market_from_symbol(self, symbol: str) -> str:
        """
        Determine market from symbol pattern.
        
        Args:
            symbol: Trading symbol (e.g., MSFT, 0001.HK, BHP.AX)
            
        Returns:
            str: Market name (us, hkex, asx)
        """
        if '.HK' in symbol:
            return 'hkex'
        elif '.AX' in symbol:
            return 'asx'
        else:
            return 'us'  # Default to US market if no suffix

    def _clean_symbol(self, symbol: str) -> Optional[str]:
        """
        Clean symbol string by removing quotes, commas, and other unwanted characters.
        
        Args:
            symbol: Raw symbol string
            
        Returns:
            Cleaned symbol string or None if invalid
        """
        try:
            if not symbol or not isinstance(symbol, str):
                return None
            
            # Remove quotes, commas, brackets, and whitespace
            clean = symbol.strip().strip('"').strip("'").strip(',').strip('[').strip(']')
            
            # Remove any remaining whitespace
            clean = clean.strip()
            
            # Skip if empty or contains invalid characters
            if not clean or clean in ['{', '}', '[', ']', 'symbols', 'symbols":', '":']:
                logger.warning(f"Invalid symbol after cleaning: '{symbol}' -> '{clean}'")
                return None
            
            # Skip if it looks like JSON syntax
            if ':' in clean or clean.startswith('{') or clean.startswith('['):
                logger.warning(f"Symbol appears to be JSON syntax: '{symbol}'")
                return None
            
            logger.debug(f"Cleaned symbol '{symbol}' to '{clean}'")
            return clean
        
        except Exception as e:
            logger.error(f"Error cleaning symbol '{symbol}': {str(e)}")
            return None

    def get_data(self, 
                 symbols: List[str], 
                 start_date: datetime, 
                 end_date: datetime,
                 timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for the specified symbols and date range.
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data timeframe (e.g., "1d", "1h", "15m")
            
        Returns:
            Dict mapping symbols to DataFrames with OHLCV data
        """
        result = {}
        
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=pytz.UTC)
        
        for symbol in symbols:
            # Clean the symbol first
            clean_symbol = self._clean_symbol(symbol)
            if not clean_symbol:
                logger.warning(f"Skipping invalid symbol: '{symbol}'")
                continue
                
            # Check if data is already in cache
            cache_key = f"{clean_symbol}_{timeframe}"
            if cache_key in self._cache:
                df = self._cache[cache_key]
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                result[clean_symbol] = df.loc[mask].copy()
                logger.info(f"Using cached data for {clean_symbol}: {len(result[clean_symbol])} rows")
                continue
            
            # Try to get data from InfluxDB first if enabled
            if self._influxdb_client:
                try:
                    df = self._get_data_from_influxdb(clean_symbol, start_date, end_date)
                    if df is not None and len(df) > 0:
                        # Add symbol column if not present
                        if 'symbol' not in df.columns:
                            df['symbol'] = clean_symbol
                        
                        # Ensure columns are properly named
                        df = self._standardize_columns(df)
                        
                        # Cache the full dataset
                        self._cache[cache_key] = df
                        
                        # Return the filtered dataset
                        mask = (df.index >= start_date) & (df.index <= end_date)
                        result[clean_symbol] = df.loc[mask].copy()
                        logger.info(f"Loaded {len(result[clean_symbol])} rows of data for {clean_symbol} from InfluxDB")
                        continue
                    else:
                        logger.warning(f"No data found in InfluxDB for {clean_symbol}, falling back to files")
                except Exception as e:
                    logger.error(f"Error retrieving data from InfluxDB for {clean_symbol}: {str(e)}")
                    logger.error(traceback.format_exc())
                    logger.warning(f"Falling back to file-based data for {clean_symbol}")
                
            # If we get here, either InfluxDB is disabled or data retrieval failed
            logger.error(f"No data source available for {clean_symbol}")
            continue
                
        return result
    
    def _get_data_from_influxdb(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get data from InfluxDB.
        
        Args:
            symbol: The formatted symbol
            start_date: Start date in UTC
            end_date: End date in UTC
            
        Returns:
            pd.DataFrame: Market data from InfluxDB
        """
        if not self._bucket_name or not self._influxdb_client or not self._query_api:
            logger.error("InfluxDB is not properly configured")
            return None
            
        try:
            # Clean up symbol for query
            clean_symbol = self._clean_symbol(symbol)
            
            # Skip invalid symbols
            if not clean_symbol:
                logger.warning(f"Invalid symbol after cleaning: '{symbol}'")
                return None
            
            logger.info(f"Querying InfluxDB for symbol: '{clean_symbol}'")
            
            # Build query with specific fields and aggregation
            query = f'''
from(bucket: "stocks")
  |> range(start: {start_date.strftime("%Y-%m-%dT%H:%M:%SZ")}, stop: {end_date.strftime("%Y-%m-%dT%H:%M:%SZ")})
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> filter(fn: (r) => r["symbol"] == "{clean_symbol}")
  |> filter(fn: (r) => r["_field"] == "close" or r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "open" or r["_field"] == "volume" or r["_field"] == "adjusted_close" or r["_field"] == "market_cap")
  |> aggregateWindow(every: 1d, fn: last, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> yield(name: "last")
'''
            
            try:
                logger.debug(f"Executing InfluxDB query for {clean_symbol}:\n{query}")
                tables = self._query_api.query_data_frame(query)
                
                if not isinstance(tables, pd.DataFrame):
                    if len(tables) == 0:
                        logger.warning(f"No data found in InfluxDB for {clean_symbol}")
                        return None
                    df = pd.concat(tables)
                else:
                    df = tables
                    
                if len(df) == 0:
                    logger.warning(f"Empty DataFrame returned for {clean_symbol}")
                    return None
                    
                # Log the raw data structure
                logger.debug(f"Raw DataFrame columns for {clean_symbol}: {df.columns.tolist()}")
                logger.debug(f"Raw DataFrame head for {clean_symbol}:\n{df.head()}")
                
                # Set index to datetime
                if '_time' in df.columns:
                    df.set_index("_time", inplace=True)
                    df.index.name = "date"
                else:
                    logger.error(f"No '_time' column found in data for {clean_symbol}")
                    logger.debug(f"Available columns: {df.columns.tolist()}")
                    return None
                
                # Drop unnecessary columns
                columns_to_drop = [col for col in df.columns if col.startswith('result') or col.startswith('table')]
                df = df.drop(columns=columns_to_drop, errors='ignore')
                
                # Drop any duplicate indices
                df = df[~df.index.duplicated(keep='first')]
                
                # Ensure all required columns exist with correct names
                required_columns = {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "adjusted_close": "adj_close",
                    "market_cap": "market_cap"
                }
                
                # Create missing columns as NaN
                for col in required_columns.keys():
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Rename columns to match expected format
                df = df.rename(columns=required_columns)
                
                # Select and order columns
                essential_columns = ["open", "high", "low", "close", "volume", "adj_close"]
                optional_columns = ["market_cap"]
                
                # Ensure essential columns exist
                df = df.reindex(columns=essential_columns + [col for col in optional_columns if col in df.columns])
                
                # Convert numeric columns
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN values in essential columns
                df = df.dropna(subset=essential_columns)
                
                # Sort index
                df = df.sort_index()
                
                if len(df) > 0:
                    logger.info(f"Successfully retrieved {len(df)} rows for {clean_symbol} from InfluxDB")
                    logger.debug(f"Final DataFrame head for {clean_symbol}:\n{df.head()}")
                    return df
                else:
                    logger.warning(f"No valid data points found for {clean_symbol} after cleaning")
                    return None
                
            except Exception as e:
                logger.error(f"Error querying InfluxDB for {clean_symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error preparing InfluxDB query for {clean_symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has standard column names and types.
        """
        # Ensure all required columns exist
        required_columns = ["open", "high", "low", "close", "volume", "adj_close"]
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Convert numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume", "adj_close"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    def _get_data_files(self, 
                       symbol: str, 
                       start_date: datetime, 
                       end_date: datetime,
                       market: str = "nasdaq") -> List[Path]:
        """
        Get all data files for a symbol within a date range.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            market: Market name
            
        Returns:
            List of paths to data files
        """
        # Construct the directory path for this symbol
        symbol_dir = self.base_path / market / symbol
        
        if not symbol_dir.exists():
            logger.warning(f"Directory not found for {symbol} in market {market}: {symbol_dir}")
            return []
        
        # Get all date files within the range
        date_files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = symbol_dir / f"{date_str}.json"
            
            if file_path.exists():
                date_files.append(file_path)
            
            current_date += timedelta(days=1)
        
        return date_files
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess market data, calculate indicators, etc.
        
        Args:
            data: Dict mapping symbols to raw market data DataFrames
            
        Returns:
            Dict mapping symbols to preprocessed DataFrames with indicators
        """
        result = {}
        
        for symbol, df in data.items():
            try:
                # Make a copy to avoid modifying the original
                processed_df = df.copy()
                
                # Calculate basic indicators
                processed_df = self._calculate_indicators(processed_df)
                
                result[symbol] = processed_df
                
            except Exception as e:
                logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
                result[symbol] = df  # Use original data if preprocessing fails
                
        return result
    
    def _load_data_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with the loaded data or None if loading fails
        """
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.json':
                # Load JSON file as a dictionary
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame (single row)
                df = pd.DataFrame([data])
                return df
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the DataFrame.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with calculated indicators
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        try:
            # Calculate moving averages
            result_df['ma_20'] = result_df['close'].rolling(window=20).mean()
            result_df['ma_50'] = result_df['close'].rolling(window=50).mean()
            result_df['ma_200'] = result_df['close'].rolling(window=200).mean()
            
            # Calculate daily returns
            result_df['daily_return'] = result_df['close'].pct_change()
            
            # Calculate exponential moving averages
            result_df['ema_12'] = result_df['close'].ewm(span=12, adjust=False).mean()
            result_df['ema_26'] = result_df['close'].ewm(span=26, adjust=False).mean()
            
            # Calculate MACD
            result_df['macd'] = result_df['ema_12'] - result_df['ema_26']
            result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
            result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
            
            # Calculate RSI (14 periods)
            delta = result_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            result_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands (20,2)
            result_df['bb_middle'] = result_df['close'].rolling(window=20).mean()
            result_df['bb_std'] = result_df['close'].rolling(window=20).std()
            result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * 2)
            result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * 2)
            
            # Calculate Average True Range (ATR)
            high_low = result_df['high'] - result_df['low']
            high_close = (result_df['high'] - result_df['close'].shift()).abs()
            low_close = (result_df['low'] - result_df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result_df['atr_14'] = true_range.rolling(14).mean()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        
        return result_df 
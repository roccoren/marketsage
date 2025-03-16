"""
Data provider for historical market data.
Handles data loading, preprocessing, and caching for backtesting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta
import logging
import os
import json
import glob

logger = logging.getLogger(__name__)

class DataProvider:
    """
    Provides historical market data for backtesting.
    Handles data loading, preprocessing, and caching.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the data provider.
        
        Args:
            base_path: Base path for market data files
        """
        self.base_path = Path(base_path)
        self.data_cache = {}
        logger.info(f"DataProvider initialized with base path: {self.base_path}")
        
    def get_data(self, 
                 symbols: List[str], 
                 start_date: datetime, 
                 end_date: datetime,
                 timeframe: str = "1d",
                 market: str = "nasdaq") -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for the specified symbols and date range.
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data timeframe (e.g., "1d", "1h", "15m")
            market: Market name (e.g., "nasdaq", "nyse")
            
        Returns:
            Dict mapping symbols to DataFrames with OHLCV data
        """
        result = {}
        
        for symbol in symbols:
            # Check if data is already in cache
            cache_key = f"{symbol}_{timeframe}_{market}"
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                result[symbol] = df.loc[mask].copy()
                continue
                
            # Get all data files for this symbol within the date range
            data_files = self._get_data_files(symbol, start_date, end_date, market)
            
            if not data_files:
                logger.warning(f"No data files found for {symbol} in market {market} between {start_date.date()} and {end_date.date()}")
                continue
                
            try:
                # Load and combine data from all files
                dfs = []
                for file_path in data_files:
                    daily_data = self._load_data_file(file_path)
                    if daily_data is not None:
                        dfs.append(daily_data)
                
                if not dfs:
                    logger.warning(f"No valid data loaded for {symbol}")
                    continue
                    
                # Combine all daily data into a single DataFrame
                df = pd.concat(dfs)
                
                # Convert date to datetime index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Add symbol column if not present
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                
                # Ensure columns are properly named
                df = self._standardize_columns(df)
                
                # Cache the full dataset
                self.data_cache[cache_key] = df
                
                # Return the filtered dataset
                mask = (df.index >= start_date) & (df.index <= end_date)
                result[symbol] = df.loc[mask].copy()
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                continue
                
        return result
    
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
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names in the DataFrame.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized column names
        """
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume',
            'Adj Close': 'adj_close',
            'Adj_Close': 'adj_close',
            'adj_close': 'adj_close',
            'AdjClose': 'adj_close',
            'adjusted_close': 'adj_close'
        }
        
        # Rename columns that exist in the mapping
        cols_to_rename = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        if cols_to_rename:
            df = df.rename(columns=cols_to_rename)
            
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = np.nan
        
        # Convert price columns to float to avoid decimal conversion issues
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in df.columns:
                # Handle any non-numeric values
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {str(e)}")
                    df[col] = df[col].astype(float)
        
        # Convert volume to int
        if 'volume' in df.columns:
            try:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"Error converting volume to int: {str(e)}")
                df['volume'] = 0
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd_12_26_9'] = df['ema_12'] - df['ema_26']
        df['macds_12_26_9'] = df['macd_12_26_9'].ewm(span=9, adjust=False).mean()
        df['macdh_12_26_9'] = df['macd_12_26_9'] - df['macds_12_26_9']
        
        # RSI (14-period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-period, 2 standard deviations)
        df['BBM_20_2.0'] = df['sma_20']
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + 2 * df['close'].rolling(window=20).std()
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - 2 * df['close'].rolling(window=20).std()
        
        # Average True Range (14-period)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        return df 
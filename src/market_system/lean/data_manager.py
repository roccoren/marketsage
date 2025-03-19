"""
LEAN Data Manager

This module provides data management functionality for the LEAN framework:
1. Fetching market data from various sources
2. Cleaning and processing data to LEAN format
3. Managing data storage and organization
4. Data validation and integrity checks
"""

import os
import pandas as pd
import numpy as np
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
import re

# Import market data components if available
try:
    from src.market_data.providers.enhanced_fetcher import EnhancedMarketFetcher
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False
    logging.warning("Market data module not found. Some data features may be limited.")

logger = logging.getLogger(__name__)

# LEAN data directories
LEAN_DATA_TYPES = {
    'equity': {
        'daily': 'equity/usa/daily',
        'minute': 'equity/usa/minute',
        'second': 'equity/usa/second',
        'tick': 'equity/usa/tick'
    },
    'forex': {
        'daily': 'forex/fxcm/daily',
        'hour': 'forex/fxcm/hour',
        'minute': 'forex/fxcm/minute',
        'second': 'forex/fxcm/second',
        'tick': 'forex/fxcm/tick'
    },
    'crypto': {
        'daily': 'crypto/coinbase/daily',
        'minute': 'crypto/coinbase/minute'
    },
    'option': {
        'daily': 'option/usa/daily',
        'minute': 'option/usa/minute'
    }
}

class LeanDataManager:
    """
    Data manager for the LEAN framework that handles data acquisition, 
    cleaning, and preparation.
    """
    
    def __init__(self, 
                 data_directory: str = "data",
                 cache_dir: str = "data/cache",
                 use_market_data_module: bool = True):
        """
        Initialize the LEAN data manager.
        
        Args:
            data_directory: Root directory for data storage
            cache_dir: Directory for caching intermediate results
            use_market_data_module: Whether to use the market data module
        """
        self.data_dir = Path(data_directory)
        self.cache_dir = Path(cache_dir)
        self.lean_data_dir = self.data_dir / "lean"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lean_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create LEAN data structure
        for asset_type, resolutions in LEAN_DATA_TYPES.items():
            for resolution, path in resolutions.items():
                (self.lean_data_dir / path).mkdir(parents=True, exist_ok=True)
        
        # Initialize market data fetcher if available
        self.market_data_available = MARKET_DATA_AVAILABLE and use_market_data_module
        if self.market_data_available:
            try:
                self.market_fetcher = EnhancedMarketFetcher(
                    config_dir="config",
                    data_dir=str(self.cache_dir),
                    use_influxdb=False
                )
                logger.info("Initialized market data fetcher")
            except Exception as e:
                logger.error(f"Failed to initialize market data fetcher: {e}")
                self.market_data_available = False
    
    async def fetch_data(self, symbols: List[str], 
                        start_date: Union[str, datetime], 
                        end_date: Optional[Union[str, datetime]] = None,
                        source: str = "yahoo",
                        market: str = "usa") -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for the specified symbols.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data (defaults to today)
            source: Data source (yahoo, alpha_vantage, etc.)
            market: Market identifier
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        results = {}
            
        # Use market data module if available
        if self.market_data_available:
            try:
                logger.info(f"Fetching data for {len(symbols)} symbols using market data module")
                
                # Fetch data for each symbol
                for symbol in symbols:
                    try:
                        # Use the market fetcher to get data
                        await self.market_fetcher.fetch_market_data(
                            market=market,
                            symbols=[symbol],
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        # Process raw data to create DataFrame
                        symbol_data = []
                        current_date = start_date
                        while current_date <= end_date:
                            data = await self.market_fetcher.get_market_data(
                                market=market,
                                symbol=symbol,
                                date=current_date
                            )
                            
                            if data:
                                symbol_data.append(data)
                            
                            current_date += timedelta(days=1)
                        
                        if symbol_data:
                            # Convert to DataFrame
                            df = pd.DataFrame(symbol_data)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            results[symbol] = df
                            logger.info(f"Successfully fetched data for {symbol}")
                        else:
                            logger.warning(f"No data fetched for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                
                return results
                
            except Exception as e:
                logger.error(f"Error using market data module: {e}")
                # Fall back to direct Yahoo Finance
        
        # Direct fetch using yfinance
        try:
            import yfinance as yf
            
            logger.info(f"Fetching data for {len(symbols)} symbols using yfinance")
            
            # Fetch all symbols at once for efficiency
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            
            # Process the data
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        # For a single symbol, yfinance doesn't use MultiIndex
                        symbol_data = data.copy()
                    else:
                        # For multiple symbols, extract the symbol's data
                        symbol_data = data[symbol].copy()
                    
                    # Clean the data
                    symbol_data = symbol_data.dropna()
                    
                    if not symbol_data.empty:
                        # Rename columns to lowercase
                        symbol_data.columns = [col.lower() for col in symbol_data.columns]
                        
                        # Ensure required columns
                        required_cols = {'open', 'high', 'low', 'close', 'volume'}
                        if not required_cols.issubset(set(symbol_data.columns)):
                            logger.warning(f"Missing required columns for {symbol}")
                            continue
                        
                        results[symbol] = symbol_data
                        logger.info(f"Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"No data fetched for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing data for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching data directly: {e}")
            return {}
    
    def clean_data(self, df: pd.DataFrame, fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Clean and prepare market data.
        
        Args:
            df: DataFrame with OHLCV data
            fill_method: Method for filling missing values ('ffill', 'bfill', 'none')
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned = df.copy()
        
        # Ensure index is datetime
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            if 'date' in cleaned.columns:
                cleaned['date'] = pd.to_datetime(cleaned['date'])
                cleaned = cleaned.set_index('date')
            elif 'timestamp' in cleaned.columns:
                cleaned['timestamp'] = pd.to_datetime(cleaned['timestamp'])
                cleaned = cleaned.set_index('timestamp')
        
        # Handle missing values
        if fill_method != 'none':
            cleaned = cleaned.fillna(method=fill_method)
        
        # Remove rows with any remaining NaN values
        cleaned = cleaned.dropna()
        
        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        
        # Ensure high >= low
        if 'high' in cleaned.columns and 'low' in cleaned.columns:
            mask = cleaned['high'] < cleaned['low']
            if mask.any():
                logger.warning(f"Found {mask.sum()} rows where high < low, fixing...")
                # Swap high and low where high < low
                temp = cleaned.loc[mask, 'high'].copy()
                cleaned.loc[mask, 'high'] = cleaned.loc[mask, 'low']
                cleaned.loc[mask, 'low'] = temp
        
        # Ensure high >= open, high >= close
        if 'high' in cleaned.columns:
            if 'open' in cleaned.columns:
                cleaned['high'] = np.maximum(cleaned['high'], cleaned['open'])
            if 'close' in cleaned.columns:
                cleaned['high'] = np.maximum(cleaned['high'], cleaned['close'])
        
        # Ensure low <= open, low <= close
        if 'low' in cleaned.columns:
            if 'open' in cleaned.columns:
                cleaned['low'] = np.minimum(cleaned['low'], cleaned['open'])
            if 'close' in cleaned.columns:
                cleaned['low'] = np.minimum(cleaned['low'], cleaned['close'])
        
        # Ensure volume is non-negative
        if 'volume' in cleaned.columns:
            cleaned.loc[cleaned['volume'] < 0, 'volume'] = 0
        
        return cleaned
    
    def convert_to_lean_format(self, 
                              df: pd.DataFrame, 
                              symbol: str, 
                              asset_type: str = 'equity',
                              resolution: str = 'daily') -> str:
        """
        Convert data to LEAN format and save to the appropriate location.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            asset_type: Asset type (equity, forex, crypto, option)
            resolution: Data resolution (daily, minute, etc.)
            
        Returns:
            Path to the saved file
        """
        # Ensure the asset type and resolution are valid
        if asset_type not in LEAN_DATA_TYPES:
            raise ValueError(f"Invalid asset type: {asset_type}")
        
        if resolution not in LEAN_DATA_TYPES[asset_type]:
            raise ValueError(f"Invalid resolution for {asset_type}: {resolution}")
        
        # Get the target directory
        target_dir = self.lean_data_dir / LEAN_DATA_TYPES[asset_type][resolution]
        target_file = target_dir / f"{symbol}.csv"
        
        # Create a copy for processing
        lean_df = df.copy()
        
        # Reset index to get date as column
        if isinstance(lean_df.index, pd.DatetimeIndex):
            lean_df = lean_df.reset_index()
            date_col = lean_df.columns[0]
        elif 'date' in lean_df.columns:
            date_col = 'date'
        elif 'timestamp' in lean_df.columns:
            date_col = 'timestamp'
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'date'/'timestamp' column")
        
        # Format according to LEAN requirements
        lean_data = pd.DataFrame()
        
        # Format the time column
        lean_data['time'] = pd.to_datetime(lean_df[date_col]).dt.strftime('%Y-%m-%d')
        
        # Add required columns
        required_cols = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        for lean_col, df_col in required_cols.items():
            if df_col in lean_df.columns:
                lean_data[lean_col] = lean_df[df_col]
            else:
                logger.warning(f"Missing column {df_col} for {symbol}")
                # Use reasonable defaults if column is missing
                if lean_col == 'volume':
                    lean_data[lean_col] = 0
                else:
                    # For OHLC, we need to have some value
                    if 'close' in lean_df.columns:
                        lean_data[lean_col] = lean_df['close']
                    else:
                        raise ValueError(f"Cannot create LEAN data without price information")
        
        # Save to file
        lean_data.to_csv(target_file, index=False)
        logger.info(f"Saved LEAN format data for {symbol} to {target_file}")
        
        return str(target_file)
    
    def prepare_lean_data(self, 
                         symbols: List[str], 
                         start_date: Union[str, datetime],
                         end_date: Optional[Union[str, datetime]] = None,
                         asset_type: str = 'equity',
                         resolution: str = 'daily',
                         market: str = 'usa') -> Dict[str, str]:
        """
        End-to-end function to fetch, clean, and prepare data for LEAN.
        
        Args:
            symbols: List of symbols to prepare
            start_date: Start date for data
            end_date: End date for data
            asset_type: Asset type (equity, forex, crypto, option)
            resolution: Data resolution (daily, minute, etc.)
            market: Market identifier
            
        Returns:
            Dictionary mapping symbols to their LEAN data file paths
        """
        loop = asyncio.get_event_loop()
        market_data = loop.run_until_complete(
            self.fetch_data(symbols, start_date, end_date, market=market)
        )
        
        results = {}
        for symbol, df in market_data.items():
            try:
                # Clean the data
                cleaned_df = self.clean_data(df)
                
                # Convert to LEAN format
                lean_path = self.convert_to_lean_format(
                    cleaned_df, 
                    symbol, 
                    asset_type=asset_type,
                    resolution=resolution
                )
                
                results[symbol] = lean_path
                
            except Exception as e:
                logger.error(f"Error preparing LEAN data for {symbol}: {e}")
        
        return results
    
    def validate_lean_data(self, symbol: str, 
                          asset_type: str = 'equity',
                          resolution: str = 'daily') -> bool:
        """
        Validate LEAN data files for a symbol.
        
        Args:
            symbol: Symbol to validate
            asset_type: Asset type (equity, forex, crypto, option)
            resolution: Data resolution (daily, minute, etc.)
            
        Returns:
            True if validation passes, False otherwise
        """
        # Check if the asset type and resolution are valid
        if asset_type not in LEAN_DATA_TYPES:
            logger.error(f"Invalid asset type: {asset_type}")
            return False
        
        if resolution not in LEAN_DATA_TYPES[asset_type]:
            logger.error(f"Invalid resolution for {asset_type}: {resolution}")
            return False
        
        # Get the file path
        file_path = self.lean_data_dir / LEAN_DATA_TYPES[asset_type][resolution] / f"{symbol}.csv"
        
        # Check if the file exists
        if not file_path.exists():
            logger.error(f"LEAN data file for {symbol} does not exist: {file_path}")
            return False
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Check columns
            required_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(set(df.columns)):
                missing = required_cols - set(df.columns)
                logger.error(f"Missing required columns in {symbol} data: {missing}")
                return False
            
            # Check data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    pd.to_numeric(df[col])
                except:
                    logger.error(f"Column {col} in {symbol} data cannot be converted to numeric")
                    return False
            
            # Check for empty data
            if df.empty:
                logger.error(f"LEAN data file for {symbol} is empty")
                return False
            
            # Check time format
            try:
                pd.to_datetime(df['time'])
            except:
                logger.error(f"Time column in {symbol} data has invalid format")
                return False
            
            # Check data values
            if (df['high'] < df['low']).any():
                logger.error(f"Found invalid price data in {symbol} (high < low)")
                return False
                
            if (df['volume'] < 0).any():
                logger.error(f"Found negative volume in {symbol} data")
                return False
            
            logger.info(f"LEAN data validation passed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating LEAN data for {symbol}: {e}")
            return False
    
    def get_available_symbols(self, 
                             asset_type: str = 'equity',
                             resolution: str = 'daily') -> List[str]:
        """
        Get a list of symbols with available LEAN data.
        
        Args:
            asset_type: Asset type (equity, forex, crypto, option)
            resolution: Data resolution (daily, minute, etc.)
            
        Returns:
            List of symbols with available data
        """
        # Check if the asset type and resolution are valid
        if asset_type not in LEAN_DATA_TYPES:
            logger.warning(f"Invalid asset type: {asset_type}")
            return []
        
        if resolution not in LEAN_DATA_TYPES[asset_type]:
            logger.warning(f"Invalid resolution for {asset_type}: {resolution}")
            return []
        
        # Get the directory
        directory = self.lean_data_dir / LEAN_DATA_TYPES[asset_type][resolution]
        
        # List CSV files
        symbols = []
        if directory.exists():
            for file in directory.glob("*.csv"):
                symbol = file.stem
                symbols.append(symbol)
        
        return symbols 
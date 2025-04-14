#!/usr/bin/env python3
"""
Data acquisition module for the Qlib-based quantitative analysis project.
This module provides functions to fetch and process stock data using Qlib's
data framework and ExpressionEngine.
"""

import os
import time
import random
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import yfinance as yf  # Import yfinance library
import qlib
from qlib.data import D
from qlib.config import C
from qlib.data.ops import *

from modules.storage import StorageManager


class DataAcquisitionManager:
    """
    Manager for data acquisition operations.
    Handles fetching and storing stock data.
    """
    
    def __init__(self, storage_manager: Optional[StorageManager] = None, retry_count: int = 3, retry_delay: int = 5):
        """
        Initialize data acquisition manager.
        
        Args:
            storage_manager (StorageManager, optional): Storage manager for data persistence
            retry_count (int): Number of retries for failed data fetching
            retry_delay (int): Delay in seconds between retries
        """
        self.storage_manager = storage_manager if storage_manager else StorageManager()
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._init_qlib()
    
    def _init_qlib(self):
        """Initialize Qlib with default settings."""
        provider_uri = os.getenv('QLIB_DATA_PATH', 'data/qlib_data')
        
        # Create data directory if it doesn't exist
        Path(provider_uri).mkdir(parents=True, exist_ok=True)
        
        # Initialize Qlib
        qlib.init(provider_uri=provider_uri,
                 region='us',  # Assuming US market
                 verbose=False)
        
    def fetch_stock_data(self, symbol: str, start_date: str = "2020-01-01", end_date: Optional[str] = None,
                        store: bool = True) -> pd.DataFrame:
        """
        Fetch historical stock data using Qlib's data framework.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            store (bool): Whether to store the fetched data
            
        Returns:
            pd.DataFrame: Historical stock data with features
        """
        try:
            # Define the basic features we want to fetch
            fields = [
                '$close', '$open', '$high', '$low', '$volume',
                'Ref($close, 1)',  # Previous close
                'Mean($close, 20)',  # 20-day moving average
                'Std($close, 20)',  # 20-day standard deviation
                'RSI($close, 14)',  # 14-day RSI
                'MACD($close)',  # MACD
            ]
            
            # Format symbol for Qlib (assuming US market symbols)
            qlib_symbol = f"{symbol.upper()}"
            
            # Fetch data using Qlib's API
            df = D.features([qlib_symbol], fields, start_time=start_date, end_time=end_date)
            
            # Rename columns to match existing format
            df = df.rename(columns={
                '$close': 'Close',
                '$open': 'Open',
                '$high': 'High',
                '$low': 'Low',
                '$volume': 'Volume',
            })
            
            if store and self.storage_manager and not df.empty:
                self.storage_manager.save_data(symbol, df)
                print(f"Stored {symbol} data with Qlib features in storage")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol} using Qlib: {e}")
            # Fallback to yfinance if Qlib fails
            return self._fetch_with_retry(symbol, start_date, end_date)
    
    def _fetch_with_retry(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data with retry logic.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data
            end_date (str, optional): End date for historical data
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        for attempt in range(self.retry_count):
            try:
                print(f"Fetching data for {symbol} (attempt {attempt + 1}/{self.retry_count})...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not df.empty:
                    return df
                    
                print(f"No data returned for {symbol}, retrying...")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
            
            # Exponential backoff with jitter
            if attempt < self.retry_count - 1:
                wait_time = min(300, (2 ** attempt) * self.retry_delay + random.uniform(1, 5))
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
        
        print(f"Failed to fetch data for {symbol} after {self.retry_count} attempts")
        return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str = "2020-01-01", 
                              end_date: Optional[str] = None, store: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stock symbols.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            store (bool): Whether to store the fetched data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data by symbol
        """
        result = {}
        
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date, store)
            if not df.empty:
                result[symbol] = df
        
        return result


class QlibFeatureEngine:
    """Feature engineering using Qlib's ExpressionEngine."""
    
    @staticmethod
    def calculate_features(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate technical features using Qlib's expression engine.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for calculation
            end_date (str, optional): End date for calculation
            
        Returns:
            pd.DataFrame: DataFrame with calculated features
        """
        # Define advanced feature expressions
        feature_expressions = {
            'momentum': 'Mean($close/Ref($close, 20)-1, 5)',
            'volatility': 'Std($close, 20)',
            'trend_strength': 'Abs(Corr($close, IndNeutralize(Range(0, 20), $close), 20))',
            'volume_price_trend': 'Corr($volume, $close, 10)',
            'bb_upper': 'Mean($close, 20) + 2 * Std($close, 20)',
            'bb_lower': 'Mean($close, 20) - 2 * Std($close, 20)',
        }
        
        try:
            # Calculate features using Qlib's expression engine
            df = D.expression(feature_expressions, instruments=[symbol],
                            start_time=start_date, end_time=end_date)
            return df
        except Exception as e:
            print(f"Error calculating features for {symbol}: {e}")
            return pd.DataFrame()

# For backward compatibility with existing code
def fetch_stock_data(symbol, start_date="2020-01-01", end_date=None):
    """
    Fetch historical stock data with Qlib features.
    Returns a pandas DataFrame.
    """
    manager = DataAcquisitionManager()
    return manager.fetch_stock_data(symbol, start_date, end_date, store=False)
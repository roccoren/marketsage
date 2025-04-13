#!/usr/bin/env python3
"""
Data acquisition module for the Qlib-based quantitative analysis project.
This module provides functions to fetch stock data from yfinance and store it
using the storage abstraction layer.
"""

import os
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

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
    
    def fetch_stock_data(self, symbol: str, start_date: str = "2020-01-01", end_date: Optional[str] = None, 
                         store: bool = True) -> pd.DataFrame:
        """
        Fetch historical stock data for the provided symbol.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            store (bool): Whether to store the fetched data
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        # Try to load from storage first if available
        if self.storage_manager and not end_date:  # Only use cached data for full history
            stored_data = self.storage_manager.load_data(symbol, start_date)
            if not stored_data.empty:
                print(f"Loaded {symbol} data from storage")
                return stored_data
        
        # Fetch from yfinance with retry logic
        df = self._fetch_with_retry(symbol, start_date, end_date)
        
        # Store the data if requested
        if store and self.storage_manager and not df.empty:
            self.storage_manager.save_data(symbol, df)
            print(f"Stored {symbol} data in storage")
        
        return df
    
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
            
            # Wait before retrying
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)
        
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


# For backward compatibility with existing code
def fetch_stock_data(symbol, start_date="2020-01-01", end_date=None):
    """
    Fetch historical stock data for the provided symbol from the start_date until today using yfinance.
    Returns a pandas DataFrame.
    
    This function is maintained for backward compatibility.
    """
    manager = DataAcquisitionManager()
    return manager.fetch_stock_data(symbol, start_date, end_date, store=False)
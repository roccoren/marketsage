#!/usr/bin/env python3
"""
Data acquisition module for fetching market data from various sources.

This module provides adapters for different data sources (Yahoo Finance, Alpha Vantage, etc.)
and a unified interface for fetching market data.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json
import csv
from pathlib import Path

from modules.trading_prediction.base import DataSource
from modules.trading_prediction.market_adapter import get_market_calendar_manager

# Configure logger
logger = logging.getLogger(__name__)


class BaseDataSource(DataSource):
    """Base implementation for data sources."""
    
    def __init__(self, name: str, cache_dir: Optional[str] = None):
        """
        Initialize BaseDataSource.
        
        Args:
            name: Name of the data source
            cache_dir: Directory for caching data (default: None)
        """
        self._name = name
        
        # Set up caching
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cache', name.lower())
        
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        """Get data source name."""
        return self._name
    
    def _get_cache_path(self, symbol: str, interval: str) -> str:
        """
        Get path for cached data.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval
            
        Returns:
            Cache file path
        """
        return os.path.join(self._cache_dir, f"{symbol}_{interval}.parquet")
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with data or None if not cached
        """
        cache_path = self._get_cache_path(symbol, interval)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Load cached data
            df = pd.read_parquet(cache_path)
            
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Check if we have all the requested data
            if len(df) > 0 and df.index.min() <= start_dt and df.index.max() >= end_dt:
                logger.info(f"Loaded {symbol} data from cache")
                return df
            
            # If we have partial data, return None to trigger a full fetch
            return None
            
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, interval: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return
        
        try:
            cache_path = self._get_cache_path(symbol, interval)
            data.to_parquet(cache_path)
            logger.info(f"Cached {symbol} data to {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _merge_with_cache(self, symbol: str, interval: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new data with cached data.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval
            new_data: New data to merge
            
        Returns:
            Merged DataFrame
        """
        cache_path = self._get_cache_path(symbol, interval)
        
        if not os.path.exists(cache_path):
            return new_data
        
        try:
            # Load cached data
            cached_data = pd.read_parquet(cache_path)
            
            # Convert index to datetime if needed
            if not isinstance(cached_data.index, pd.DatetimeIndex):
                cached_data.index = pd.to_datetime(cached_data.index)
            
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)
            
            # Combine data
            combined = pd.concat([cached_data, new_data])
            
            # Remove duplicates
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Sort by date
            combined = combined.sort_index()
            
            return combined
            
        except Exception as e:
            logger.warning(f"Error merging with cache: {e}")
            return new_data
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
            
        Returns:
            DataFrame with historical data (datetime index, OHLCV columns)
        """
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol, start_date, end_date, interval)
        if cached_data is not None:
            return cached_data
        
        # Fetch data from source
        data = self._fetch_data_from_source(symbol, start_date, end_date, interval)
        
        if data is not None and not data.empty:
            # Merge with cache and save
            merged_data = self._merge_with_cache(symbol, interval, data)
            self._save_to_cache(symbol, interval, merged_data)
            
            # Filter to requested date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            filtered_data = merged_data[(merged_data.index >= start_dt) & (merged_data.index <= end_dt)]
            
            return filtered_data
        
        return pd.DataFrame()
    
    @abstractmethod
    def _fetch_data_from_source(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from the source.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        pass


class YahooFinanceDataSource(BaseDataSource):
    """Data source for Yahoo Finance."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize Yahoo Finance data source."""
        super().__init__("YahooFinance", cache_dir)
        
        # Import yfinance here to avoid importing it at module level
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.error("yfinance package not installed. Please install it with: pip install yfinance")
            self.yf = None
    
    def _fetch_data_from_source(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        if self.yf is None:
            logger.error("yfinance package not available")
            return pd.DataFrame()
        
        try:
            # Map interval to yfinance format
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "1d": "1d",
                "5d": "5d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo"
            }
            
            yf_interval = interval_map.get(interval, "1d")
            
            # Fetch data
            data = self.yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False,
                auto_adjust=True  # Adjust OHLC automatically
            )
            
            # Handle multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Ensure we have the expected columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in expected_columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} missing for {symbol}, adding empty column")
                    data[col] = np.nan
            
            # Ensure index is timezone-naive
            if hasattr(data.index, "tz_convert") and data.index.tz is not None:
                data.index = data.index.tz_convert(None)
            
            # Normalize index to datetime
            data.index = pd.to_datetime(data.index).normalize()
            
            # Sort by date
            data = data.sort_index()
            
            # Add symbol as attribute
            data.name = symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of available ticker symbols
        """
        # Yahoo Finance doesn't provide a direct API for listing all available symbols
        # This would typically be implemented using a pre-defined list or external source
        logger.warning("get_available_symbols not fully implemented for Yahoo Finance")
        return []


class AlphaVantageDataSource(BaseDataSource):
    """Data source for Alpha Vantage."""
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize Alpha Vantage data source.
        
        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory for caching data
        """
        super().__init__("AlphaVantage", cache_dir)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def _fetch_data_from_source(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Map interval to Alpha Vantage function
            function = "TIME_SERIES_DAILY_ADJUSTED"  # Default
            av_interval = "Daily"
            
            if interval == "1m":
                function = "TIME_SERIES_INTRADAY"
                av_interval = "1min"
            elif interval == "5m":
                function = "TIME_SERIES_INTRADAY"
                av_interval = "5min"
            elif interval == "15m":
                function = "TIME_SERIES_INTRADAY"
                av_interval = "15min"
            elif interval == "30m":
                function = "TIME_SERIES_INTRADAY"
                av_interval = "30min"
            elif interval == "1h":
                function = "TIME_SERIES_INTRADAY"
                av_interval = "60min"
            elif interval == "1d":
                function = "TIME_SERIES_DAILY_ADJUSTED"
                av_interval = "Daily"
            elif interval == "1wk":
                function = "TIME_SERIES_WEEKLY_ADJUSTED"
                av_interval = "Weekly"
            elif interval == "1mo":
                function = "TIME_SERIES_MONTHLY_ADJUSTED"
                av_interval = "Monthly"
            
            # Prepare request parameters
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            # Add interval parameter for intraday data
            if function == "TIME_SERIES_INTRADAY":
                params["interval"] = av_interval
            
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract time series data
            time_series_key = f"Time Series ({av_interval})"
            if function == "TIME_SERIES_WEEKLY_ADJUSTED":
                time_series_key = "Weekly Adjusted Time Series"
            elif function == "TIME_SERIES_MONTHLY_ADJUSTED":
                time_series_key = "Monthly Adjusted Time Series"
            
            if time_series_key not in data:
                logger.error(f"Error in Alpha Vantage response: {data}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            
            # Rename columns
            column_map = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adjusted Close",
                "5. volume": "Volume",
                "6. volume": "Volume"
            }
            df = df.rename(columns=column_map)
            
            # Convert values to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Add symbol as attribute
            df.name = symbol
            
            # Ensure we have the expected columns
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in expected_columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} missing for {symbol}, adding empty column")
                    df[col] = np.nan
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of available ticker symbols
        """
        try:
            # Alpha Vantage provides a list of supported symbols
            params = {
                "function": "LISTING_STATUS",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # The response is a CSV file
            symbols = []
            lines = response.text.strip().split('\n')
            reader = csv.reader(lines)
            next(reader)  # Skip header
            
            for row in reader:
                if row:
                    symbols.append(row[0])  # Symbol is the first column
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching available symbols from Alpha Vantage: {e}")
            return []


class DataManager:
    """Manager for data acquisition from multiple sources."""
    
    def __init__(self):
        """Initialize DataManager."""
        self._sources = {}
        self._default_source = None
    
    def register_source(self, source: DataSource, is_default: bool = False) -> None:
        """
        Register a data source.
        
        Args:
            source: DataSource instance
            is_default: Whether this is the default source
        """
        self._sources[source.name] = source
        
        if is_default or self._default_source is None:
            self._default_source = source.name
    
    def get_source(self, name: str) -> DataSource:
        """
        Get a data source by name.
        
        Args:
            name: Data source name
            
        Returns:
            DataSource instance
            
        Raises:
            KeyError: If source is not registered
        """
        if name not in self._sources:
            raise KeyError(f"Data source not registered: {name}")
        
        return self._sources[name]
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                  interval: str = "1d", source: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            source: Data source name (default: use default source)
            
        Returns:
            DataFrame with historical data
        """
        if source is None:
            source = self._default_source
        
        if source not in self._sources:
            raise KeyError(f"Data source not registered: {source}")
        
        return self._sources[source].fetch_data(symbol, start_date, end_date, interval)
    
    def fetch_data_multi_source(self, symbol: str, start_date: str, end_date: str,
                               interval: str = "1d", sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Try to fetch data from multiple sources until successful.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            sources: List of data source names to try (default: all registered sources)
            
        Returns:
            DataFrame with historical data
        """
        if sources is None:
            sources = list(self._sources.keys())
        
        for source_name in sources:
            try:
                data = self.fetch_data(symbol, start_date, end_date, interval, source_name)
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {symbol} data from {source_name}")
                    return data
            except Exception as e:
                logger.warning(f"Error fetching {symbol} from {source_name}: {e}")
        
        logger.error(f"Failed to fetch {symbol} data from any source")
        return pd.DataFrame()


# Create a global instance of the data manager
_data_manager = None

def get_data_manager() -> DataManager:
    """
    Get the global data manager instance.
    
    Returns:
        DataManager instance
    """
    global _data_manager
    
    if _data_manager is None:
        _data_manager = DataManager()
        
        # Register default data sources
        try:
            _data_manager.register_source(YahooFinanceDataSource(), is_default=True)
        except Exception as e:
            logger.warning(f"Failed to register Yahoo Finance data source: {e}")
    
    return _data_manager
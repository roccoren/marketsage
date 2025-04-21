#!/usr/bin/env python3
"""
Data acquisition module for the MarketSage DataHub component.
This module provides functions to fetch and process stock data from various sources.
"""

import os
import time
import random
import logging
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup logging
logger = logging.getLogger(__name__)


class DataSource:
    """Base class for data sources."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data source.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the data source
        """
        self.config = config or {}
    
    def fetch_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        raise NotImplementedError("Subclasses must implement fetch_data method")


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception)),
        reraise=True
    )
    def fetch_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        logger.info(f"Fetching data for {symbol} from Yahoo Finance...")
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Handle multi-level column format if present
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = [col[0] for col in df.columns]
            
            # Ensure we have the expected column names
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in expected_columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} missing for {symbol}, adding empty column")
                    df[col] = 0.0
            
            # Set the name attribute for identification
            df.name = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source."""
    
    def fetch_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        # This is a placeholder for Alpha Vantage implementation
        # In a real implementation, this would use the Alpha Vantage API
        logger.info(f"Alpha Vantage source not fully implemented yet")
        return pd.DataFrame()


class DataAcquisition:
    """
    Manager for data acquisition operations.
    Handles fetching and storing stock data from various sources.
    """
    
    def __init__(self, storage_manager=None, config: Dict[str, Any] = None):
        """
        Initialize data acquisition manager.
        
        Args:
            storage_manager: Storage manager for data persistence
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}
        self.storage_manager = storage_manager
        self.acquisition_config = self.config.get("acquisition", {})
        
        # Configure retry settings
        retry_config = self.acquisition_config.get("retry", {})
        self.retry_count = retry_config.get("max_attempts", 3)
        self.retry_delay = retry_config.get("backoff_factor", 2)
        
        # Initialize data sources
        self.sources = self._init_data_sources()
        
        # Default source
        self.default_source = "yahoo"
    
    def _init_data_sources(self) -> Dict[str, DataSource]:
        """
        Initialize data sources based on configuration.
        
        Returns:
            Dict[str, DataSource]: Dictionary of data sources
        """
        sources = {}
        sources_config = self.acquisition_config.get("sources", {})
        
        # Initialize Yahoo Finance source if enabled
        if sources_config.get("yahoo", {}).get("enabled", True):
            sources["yahoo"] = YahooFinanceSource(config=sources_config.get("yahoo", {}))
        
        # Initialize Alpha Vantage source if enabled
        if sources_config.get("alpha_vantage", {}).get("enabled", False):
            sources["alpha_vantage"] = AlphaVantageSource(config=sources_config.get("alpha_vantage", {}))
        
        return sources
    
    def download_data(self, symbols: List[str], start_date: str = "2020-01-01", 
                     end_date: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Download historical data for multiple stock symbols.
        
        Args:
            symbols (List[str]): List of stock symbols
            start_date (str): Start date for historical data (YYYY-MM-DD)
            end_date (str, optional): End date for historical data (YYYY-MM-DD)
            source (str, optional): Data source to use
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of download results by symbol
        """
        results = {}
        source_name = source or self.default_source
        
        if source_name not in self.sources:
            logger.error(f"Data source '{source_name}' not available")
            return {symbol: {"success": False, "error": f"Data source '{source_name}' not available"} for symbol in symbols}
        
        data_source = self.sources[source_name]
        
        # Get max concurrent downloads from config
        max_workers = self.acquisition_config.get("max_concurrent_downloads", 5)
        
        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_symbol = {
                executor.submit(self._download_symbol_data, symbol, data_source, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Error downloading data for {symbol}: {e}")
                    results[symbol] = {"success": False, "error": str(e)}
        
        return results
    
    def _download_symbol_data(self, symbol: str, data_source: DataSource, 
                             start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Download data for a single symbol.
        
        Args:
            symbol (str): Stock symbol
            data_source (DataSource): Data source to use
            start_date (str): Start date for historical data
            end_date (str, optional): End date for historical data
            
        Returns:
            Dict[str, Any]: Download result
        """
        try:
            # Fetch data
            df = data_source.fetch_data(symbol, start_date, end_date)
            
            if df.empty:
                return {"success": False, "error": "No data returned"}
            
            # Store data if storage manager is available
            if self.storage_manager:
                self.storage_manager.save_data(symbol, df)
                logger.info(f"Stored {symbol} data in storage")
            
            return {
                "success": True,
                "rows": len(df),
                "start_date": df.index.min().strftime("%Y-%m-%d") if not df.empty else None,
                "end_date": df.index.max().strftime("%Y-%m-%d") if not df.empty else None,
            }
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.
        
        Returns:
            List[str]: List of available data source names
        """
        return list(self.sources.keys())
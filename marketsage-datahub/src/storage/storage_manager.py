#!/usr/bin/env python3
"""
Storage abstraction layer for the MarketSage DataHub component.
This module provides a unified interface for storing and retrieving data
from different storage backends (CSV, SQLite, Time-Series DB via gRPC).
"""

import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save data for a symbol."""
        pass
    
    @abstractmethod
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load data for a symbol with optional date filtering."""
        pass
    
    @abstractmethod
    def list_symbols(self) -> List[str]:
        """List all available symbols."""
        pass


class CSVStorageBackend(StorageBackend):
    """CSV file storage backend."""
    
    def __init__(self, base_dir: str = "data/csv/stock_data"):
        """
        Initialize CSV storage backend.
        
        Args:
            base_dir (str): Base directory for CSV files
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol to a CSV file.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        try:
            # Create file path
            file_path = os.path.join(self.base_dir, f"{symbol}.csv")
            
            # Save to CSV
            data.to_csv(file_path)
            logger.info(f"Saved data for {symbol} to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol from a CSV file.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Create file path
            file_path = os.path.join(self.base_dir, f"{symbol}.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"No data file found for {symbol}")
                return pd.DataFrame()
            
            # Load from CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Apply date filtering if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        try:
            # List CSV files in the base directory
            files = os.listdir(self.base_dir)
            
            # Extract symbol names from file names
            symbols = [os.path.splitext(f)[0] for f in files if f.endswith('.csv')]
            
            return symbols
        except Exception as e:
            logger.error(f"Error listing symbols: {e}")
            return []


class TimeSeriesDBBackend(StorageBackend):
    """Time Series Database backend using gRPC."""
    
    def __init__(self, grpc_host: str = "localhost", grpc_port: int = 50051, timeout: int = 30):
        """
        Initialize Time Series DB backend.
        
        Args:
            grpc_host (str): gRPC server host
            grpc_port (int): gRPC server port
            timeout (int): Request timeout in seconds
        """
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.timeout = timeout
        
        # This is a placeholder for the gRPC client
        # In a real implementation, this would initialize the gRPC client
        logger.info(f"Initialized Time Series DB backend with gRPC at {grpc_host}:{grpc_port}")
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol to the Time Series DB.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        try:
            # This is a placeholder for the gRPC client call
            # In a real implementation, this would call the gRPC service
            logger.info(f"Saving data for {symbol} to Time Series DB (rows: {len(data)})")
            
            # Simulate success
            return True
        except Exception as e:
            logger.error(f"Error saving data for {symbol} to Time Series DB: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol from the Time Series DB.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # This is a placeholder for the gRPC client call
            # In a real implementation, this would call the gRPC service
            logger.info(f"Loading data for {symbol} from Time Series DB")
            
            # Simulate empty result
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data for {symbol} from Time Series DB: {e}")
            return pd.DataFrame()
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        try:
            # This is a placeholder for the gRPC client call
            # In a real implementation, this would call the gRPC service
            logger.info("Listing symbols from Time Series DB")
            
            # Simulate empty result
            return []
        except Exception as e:
            logger.error(f"Error listing symbols from Time Series DB: {e}")
            return []


class StorageManager:
    """
    Storage manager that provides a unified interface for different storage backends.
    Supports both synchronous and asynchronous operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize storage manager with the specified backend.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}
        self.storage_config = self.config.get("storage", {})
        
        # Initialize backend
        self.backend_type = self._get_default_backend_type()
        self.backend = self._create_backend(self.backend_type)
        
        # Initialize asyncio loop for async operations
        self.loop = asyncio.new_event_loop()
        
        logger.info(f"Initialized StorageManager with {self.backend_type} backend")
    
    def _get_default_backend_type(self) -> str:
        """
        Get the default backend type from configuration.
        
        Returns:
            str: Default backend type
        """
        # Check if TSDS is enabled
        if self.storage_config.get("tsds", {}).get("enabled", False):
            return "tsdb"
        
        # Default to CSV
        return "csv"
    
    def _create_backend(self, backend_type: str) -> StorageBackend:
        """
        Create a storage backend based on the specified type.
        
        Args:
            backend_type (str): Backend type
            
        Returns:
            StorageBackend: Storage backend instance
        """
        if backend_type == "csv":
            # Get CSV directory from config or use default
            csv_dir = self.storage_config.get("csv", {}).get("directory", "data/csv/stock_data")
            return CSVStorageBackend(base_dir=csv_dir)
        
        elif backend_type == "tsdb":
            # Get TSDS configuration
            tsds_config = self.storage_config.get("tsds", {})
            grpc_host = tsds_config.get("grpc_host", "localhost")
            grpc_port = tsds_config.get("grpc_port", 50051)
            timeout = tsds_config.get("timeout", 30)
            
            return TimeSeriesDBBackend(
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                timeout=timeout
            )
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def switch_backend(self, backend_type: str) -> None:
        """
        Switch to a different storage backend.
        
        Args:
            backend_type (str): Backend type
        """
        self.backend_type = backend_type
        self.backend = self._create_backend(backend_type)
        logger.info(f"Switched to {backend_type} backend")
    
    # Synchronous methods
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        return self.backend.save_data(symbol, data)
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol with optional date filtering.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return self.backend.load_data(symbol, start_date, end_date)
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        return self.backend.list_symbols()
    
    # Asynchronous methods
    
    async def save_data_async(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol asynchronously.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        return await self.loop.run_in_executor(None, self.backend.save_data, symbol, data)
    
    async def load_data_async(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol asynchronously.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return await self.loop.run_in_executor(None, self.backend.load_data, symbol, start_date, end_date)
    
    async def list_symbols_async(self) -> List[str]:
        """
        List all available symbols asynchronously.
        
        Returns:
            List[str]: List of available symbols
        """
        return await self.loop.run_in_executor(None, self.backend.list_symbols)
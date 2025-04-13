#!/usr/bin/env python3
"""
Storage abstraction layer for the Qlib-based quantitative analysis project.
This module provides a unified interface for storing and retrieving data
from different storage backends (CSV, SQLite, Time-Series DB).
"""

import os
import json
import sqlite3
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd


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
    def save_model_params(self, symbol: str, model_name: str, params: Dict[str, Any]) -> bool:
        """Save model parameters."""
        pass
    
    @abstractmethod
    def load_model_params(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Load model parameters."""
        pass
    
    @abstractmethod
    def list_symbols(self) -> List[str]:
        """List all available symbols."""
        pass
    
    @abstractmethod
    def list_models(self, symbol: str) -> List[str]:
        """List all available models for a symbol."""
        pass


class StorageManager:
    """
    Storage manager that provides a unified interface for different storage backends.
    Supports both synchronous and asynchronous operations.
    """
    
    def __init__(self, backend_type: str = "csv", **backend_kwargs):
        """Initialize storage manager with the specified backend."""
        self.backend_type = backend_type
        self.backend = self._create_backend(backend_type, **backend_kwargs)
        self.loop = asyncio.new_event_loop()
    
    def _create_backend(self, backend_type: str, **kwargs) -> StorageBackend:
        """Create a storage backend based on the specified type."""
        if backend_type == "csv":
            from modules.storage_backends.csv_storage import CSVStorage
            return CSVStorage(**kwargs)
        elif backend_type == "sqlite":
            from modules.storage_backends.sqlite_storage import SQLiteStorage
            return SQLiteStorage(**kwargs)
        elif backend_type == "tsdb":
            from modules.storage_backends.tsdb_storage import TimeSeriesDBStorage
            return TimeSeriesDBStorage(**kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def switch_backend(self, backend_type: str, **backend_kwargs) -> None:
        """Switch to a different storage backend."""
        self.backend_type = backend_type
        self.backend = self._create_backend(backend_type, **backend_kwargs)
    
    # Synchronous methods
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save data for a symbol."""
        return self.backend.save_data(symbol, data)
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load data for a symbol with optional date filtering."""
        return self.backend.load_data(symbol, start_date, end_date)
    
    def save_model_params(self, symbol: str, model_name: str, params: Dict[str, Any]) -> bool:
        """Save model parameters."""
        return self.backend.save_model_params(symbol, model_name, params)
    
    def load_model_params(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Load model parameters."""
        return self.backend.load_model_params(symbol, model_name)
    
    def list_symbols(self) -> List[str]:
        """List all available symbols."""
        return self.backend.list_symbols()
    
    def list_models(self, symbol: str) -> List[str]:
        """List all available models for a symbol."""
        return self.backend.list_models(symbol)
    
    # Asynchronous methods
    
    async def save_data_async(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save data for a symbol asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.save_data, symbol, data)
    
    async def load_data_async(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load data for a symbol asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.load_data, symbol, start_date, end_date)
    
    async def save_model_params_async(self, symbol: str, model_name: str, params: Dict[str, Any]) -> bool:
        """Save model parameters asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.save_model_params, symbol, model_name, params)
    
    async def load_model_params_async(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Load model parameters asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.load_model_params, symbol, model_name)
    
    async def list_symbols_async(self) -> List[str]:
        """List all available symbols asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.list_symbols)
    
    async def list_models_async(self, symbol: str) -> List[str]:
        """List all available models for a symbol asynchronously."""
        return await self.loop.run_in_executor(None, self.backend.list_models, symbol)
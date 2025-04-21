#!/usr/bin/env python3
"""
Unit tests for the StorageManager module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsage_datahub.storage import StorageManager, StorageBackend


class TestStorageManager(unittest.TestCase):
    """Test cases for the StorageManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create a test configuration
        self.config = {
            "storage": {
                "csv": {
                    "directory": self.data_dir
                },
                "tsds": {
                    "enabled": False,
                    "grpc_host": "localhost",
                    "grpc_port": 50051
                }
            }
        }
        
        # Create test data
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [98.0, 99.0, 100.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_init_with_csv_backend(self):
        """Test initialization with CSV backend."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Verify the backend type
        self.assertEqual(manager.backend_type, "csv")
        self.assertIsInstance(manager.backend, StorageBackend)
    
    def test_init_with_tsds_backend(self):
        """Test initialization with TSDS backend."""
        # Update config to enable TSDS
        config = self.config.copy()
        config["storage"]["tsds"]["enabled"] = True
        
        # Create the storage manager
        manager = StorageManager(config=config)
        
        # Verify the backend type
        self.assertEqual(manager.backend_type, "tsdb")
        self.assertIsInstance(manager.backend, StorageBackend)
    
    def test_switch_backend(self):
        """Test switching backends."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Verify initial backend
        self.assertEqual(manager.backend_type, "csv")
        
        # Switch to TSDB backend
        manager.switch_backend("tsdb")
        
        # Verify the backend was switched
        self.assertEqual(manager.backend_type, "tsdb")
        self.assertIsInstance(manager.backend, StorageBackend)
        
        # Test switching to an invalid backend
        with self.assertRaises(ValueError):
            manager.switch_backend("invalid")
    
    def test_save_and_load_data_csv(self):
        """Test saving and loading data with CSV backend."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Save test data
        symbol = "AAPL"
        result = manager.save_data(symbol, self.test_data)
        
        # Verify the save was successful
        self.assertTrue(result)
        
        # Verify the file was created
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        self.assertTrue(os.path.exists(file_path))
        
        # Load the data
        loaded_data = manager.load_data(symbol)
        
        # Verify the loaded data
        self.assertFalse(loaded_data.empty)
        self.assertEqual(loaded_data.shape, self.test_data.shape)
        self.assertTrue((loaded_data.columns == self.test_data.columns).all())
    
    def test_load_data_with_date_filtering(self):
        """Test loading data with date filtering."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Save test data
        symbol = "AAPL"
        manager.save_data(symbol, self.test_data)
        
        # Load with start date
        loaded_data = manager.load_data(symbol, start_date="2023-01-02")
        
        # Verify the loaded data
        self.assertEqual(len(loaded_data), 2)
        
        # Load with end date
        loaded_data = manager.load_data(symbol, end_date="2023-01-02")
        
        # Verify the loaded data
        self.assertEqual(len(loaded_data), 2)
        
        # Load with both start and end date
        loaded_data = manager.load_data(symbol, start_date="2023-01-02", end_date="2023-01-02")
        
        # Verify the loaded data
        self.assertEqual(len(loaded_data), 1)
    
    def test_load_nonexistent_data(self):
        """Test loading data that doesn't exist."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Load nonexistent data
        loaded_data = manager.load_data("NONEXISTENT")
        
        # Verify an empty DataFrame was returned
        self.assertTrue(loaded_data.empty)
    
    def test_list_symbols(self):
        """Test listing symbols."""
        # Create the storage manager
        manager = StorageManager(config=self.config)
        
        # Save test data for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            manager.save_data(symbol, self.test_data)
        
        # List symbols
        listed_symbols = manager.list_symbols()
        
        # Verify the symbols were listed
        self.assertEqual(len(listed_symbols), 3)
        for symbol in symbols:
            self.assertIn(symbol, listed_symbols)
    
    @patch('asyncio.new_event_loop')
    def test_async_methods(self, mock_loop):
        """Test asynchronous methods."""
        # Create a mock loop
        mock_future = MagicMock()
        mock_future.result.return_value = self.test_data
        mock_loop.run_in_executor.return_value = mock_future
        
        # Create the storage manager with the mock loop
        manager = StorageManager(config=self.config)
        manager.loop = mock_loop
        
        # Mock the backend methods
        manager.backend.load_data = MagicMock(return_value=self.test_data)
        manager.backend.save_data = MagicMock(return_value=True)
        manager.backend.list_symbols = MagicMock(return_value=["AAPL", "MSFT"])
        
        # Test async load_data
        import asyncio
        asyncio.run(manager.load_data_async("AAPL"))
        mock_loop.run_in_executor.assert_called()
        
        # Test async save_data
        asyncio.run(manager.save_data_async("AAPL", self.test_data))
        mock_loop.run_in_executor.assert_called()
        
        # Test async list_symbols
        asyncio.run(manager.list_symbols_async())
        mock_loop.run_in_executor.assert_called()


if __name__ == '__main__':
    unittest.main()
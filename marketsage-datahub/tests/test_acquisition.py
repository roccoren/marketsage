#!/usr/bin/env python3
"""
Unit tests for the DataAcquisition module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsage_datahub.acquisition import DataAcquisition
from marketsage_datahub.storage import StorageManager


class TestDataAcquisition(unittest.TestCase):
    """Test cases for the DataAcquisition class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock storage manager
        self.mock_storage = MagicMock(spec=StorageManager)
        
        # Create a test configuration
        self.config = {
            "acquisition": {
                "retry": {
                    "max_attempts": 2,
                    "backoff_factor": 1
                },
                "sources": {
                    "yahoo": {
                        "enabled": True
                    },
                    "alpha_vantage": {
                        "enabled": False
                    }
                }
            }
        }
        
        # Create the data acquisition instance
        self.acquisition = DataAcquisition(
            storage_manager=self.mock_storage,
            config=self.config
        )
    
    @patch('marketsage_datahub.acquisition.data_acquisition.yf.download')
    def test_download_data_success(self, mock_download):
        """Test successful data download."""
        # Create a sample DataFrame to return from the mock
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [98.0, 99.0, 100.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Configure the mock to return the sample data
        mock_download.return_value = sample_data
        
        # Call the method under test
        result = self.acquisition.download_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Verify the results
        self.assertTrue("AAPL" in result)
        self.assertTrue(result["AAPL"]["success"])
        self.assertEqual(result["AAPL"]["rows"], 3)
        
        # Verify the mock was called correctly
        mock_download.assert_called_once_with(
            "AAPL",
            start="2023-01-01",
            end="2023-01-03",
            progress=False
        )
        
        # Verify storage was called
        self.mock_storage.save_data.assert_called_once()
    
    @patch('marketsage_datahub.acquisition.data_acquisition.yf.download')
    def test_download_data_empty(self, mock_download):
        """Test download with empty result."""
        # Configure the mock to return an empty DataFrame
        mock_download.return_value = pd.DataFrame()
        
        # Call the method under test
        result = self.acquisition.download_data(
            symbols=["INVALID"],
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Verify the results
        self.assertTrue("INVALID" in result)
        self.assertFalse(result["INVALID"]["success"])
        self.assertEqual(result["INVALID"]["error"], "No data returned")
        
        # Verify the mock was called correctly
        mock_download.assert_called_once()
        
        # Verify storage was not called
        self.mock_storage.save_data.assert_not_called()
    
    @patch('marketsage_datahub.acquisition.data_acquisition.yf.download')
    def test_download_data_exception(self, mock_download):
        """Test download with exception."""
        # Configure the mock to raise an exception
        mock_download.side_effect = Exception("API error")
        
        # Call the method under test
        result = self.acquisition.download_data(
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-03"
        )
        
        # Verify the results
        self.assertTrue("AAPL" in result)
        self.assertFalse(result["AAPL"]["success"])
        self.assertEqual(result["AAPL"]["error"], "API error")
        
        # Verify the mock was called correctly
        mock_download.assert_called()
        
        # Verify storage was not called
        self.mock_storage.save_data.assert_not_called()
    
    def test_get_available_sources(self):
        """Test getting available data sources."""
        sources = self.acquisition.get_available_sources()
        self.assertEqual(len(sources), 1)
        self.assertIn("yahoo", sources)
        self.assertNotIn("alpha_vantage", sources)


if __name__ == '__main__':
    unittest.main()
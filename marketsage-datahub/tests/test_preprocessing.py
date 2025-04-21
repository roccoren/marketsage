#!/usr/bin/env python3
"""
Unit tests for the DataPreprocessor module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsage_datahub.preprocessing import DataPreprocessor
from marketsage_datahub.preprocessing.data_preprocessor import (
    RemoveOutliers,
    FillMissingValues,
    Normalize,
    CalculateTechnicalIndicators,
    PreprocessingStep
)


class TestPreprocessingSteps(unittest.TestCase):
    """Test cases for individual preprocessing steps."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data with outliers and missing values
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 150.0, np.nan],
            'High': [105.0, 106.0, 107.0, 155.0, 108.0],
            'Low': [98.0, 99.0, 100.0, 145.0, 99.0],
            'Close': [103.0, 104.0, 105.0, 152.0, 103.0],
            'Volume': [1000, 1100, 1200, 5000, 1100]
        }, index=pd.date_range('2023-01-01', periods=5))
    
    def test_remove_outliers(self):
        """Test removing outliers."""
        # Create the step
        step = RemoveOutliers(method="z_score", threshold=2.0)
        
        # Process the data
        result = step.process(self.test_data)
        
        # Verify outliers were removed (replaced with NaN)
        self.assertTrue(pd.isna(result.iloc[3, 0]))  # Open
        self.assertTrue(pd.isna(result.iloc[3, 1]))  # High
        self.assertTrue(pd.isna(result.iloc[3, 2]))  # Low
        self.assertTrue(pd.isna(result.iloc[3, 3]))  # Close
        self.assertTrue(pd.isna(result.iloc[3, 4]))  # Volume
        
        # Verify other values were preserved
        self.assertEqual(result.iloc[0, 0], 100.0)
        self.assertEqual(result.iloc[1, 0], 101.0)
    
    def test_fill_missing_values(self):
        """Test filling missing values."""
        # Create the step
        step = FillMissingValues(method="interpolate", limit=5)
        
        # Process the data
        result = step.process(self.test_data)
        
        # Verify missing values were filled
        self.assertFalse(pd.isna(result.iloc[4, 0]))  # Open
        
        # Verify the filled value is reasonable (interpolated)
        self.assertTrue(100.0 < result.iloc[4, 0] < 110.0)
    
    def test_normalize(self):
        """Test normalization."""
        # Create the step
        step = Normalize(method="minmax")
        
        # Process the data
        result = step.process(self.test_data)
        
        # Verify the data was normalized to [0, 1]
        for col in result.columns:
            non_na_values = result[col].dropna()
            if len(non_na_values) > 0:
                self.assertGreaterEqual(non_na_values.min(), 0.0)
                self.assertLessEqual(non_na_values.max(), 1.0)
    
    def test_calculate_technical_indicators(self):
        """Test calculating technical indicators."""
        # Create larger test data for indicators
        dates = pd.date_range('2023-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 30),
            'High': np.random.uniform(105, 115, 30),
            'Low': np.random.uniform(95, 105, 30),
            'Close': np.random.uniform(100, 110, 30),
            'Volume': np.random.uniform(1000, 2000, 30)
        }, index=dates)
        
        # Create the step
        step = CalculateTechnicalIndicators(indicators=['sma', 'ema', 'rsi', 'macd'])
        
        # Process the data
        result = step.process(data)
        
        # Verify indicators were calculated
        self.assertIn('SMA_20', result.columns)
        self.assertIn('EMA_20', result.columns)
        self.assertIn('RSI_14', result.columns)
        self.assertIn('MACD', result.columns)
        
        # Verify indicator values are not all NaN
        self.assertTrue(result['SMA_20'].notna().any())
        self.assertTrue(result['EMA_20'].notna().any())
        self.assertTrue(result['RSI_14'].notna().any())
        self.assertTrue(result['MACD'].notna().any())


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock storage manager
        self.mock_storage = MagicMock()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 150.0, np.nan],
            'High': [105.0, 106.0, 107.0, 155.0, 108.0],
            'Low': [98.0, 99.0, 100.0, 145.0, 99.0],
            'Close': [103.0, 104.0, 105.0, 152.0, 103.0],
            'Volume': [1000, 1100, 1200, 5000, 1100]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Configure mock storage manager
        self.mock_storage.load_data.return_value = self.test_data
        self.mock_storage.save_data.return_value = True
        
        # Create a test configuration
        self.config = {
            "preprocessing": {
                "default_steps": ["remove_outliers", "fill_missing_values"],
                "outlier_detection": {
                    "method": "z_score",
                    "threshold": 2.0
                },
                "missing_values": {
                    "method": "interpolate",
                    "limit": 5
                },
                "normalization": {
                    "method": "minmax"
                }
            }
        }
        
        # Create the preprocessor
        self.preprocessor = DataPreprocessor(
            storage_manager=self.mock_storage,
            config=self.config
        )
    
    def test_init_preprocessing_steps(self):
        """Test initialization of preprocessing steps."""
        # Verify the steps were initialized
        self.assertIn("remove_outliers", self.preprocessor.steps)
        self.assertIn("fill_missing_values", self.preprocessor.steps)
        self.assertIn("normalize", self.preprocessor.steps)
        self.assertIn("calculate_indicators", self.preprocessor.steps)
        
        # Verify the steps are of the correct type
        self.assertIsInstance(self.preprocessor.steps["remove_outliers"], RemoveOutliers)
        self.assertIsInstance(self.preprocessor.steps["fill_missing_values"], FillMissingValues)
        self.assertIsInstance(self.preprocessor.steps["normalize"], Normalize)
        self.assertIsInstance(self.preprocessor.steps["calculate_indicators"], CalculateTechnicalIndicators)
    
    def test_preprocess_data(self):
        """Test preprocessing data."""
        # Preprocess data
        result = self.preprocessor.preprocess_data(["AAPL"])
        
        # Verify the result
        self.assertIn("AAPL", result)
        self.assertTrue(result["AAPL"]["success"])
        self.assertEqual(result["AAPL"]["rows"], 5)
        self.assertEqual(len(result["AAPL"]["applied_steps"]), 2)
        
        # Verify the storage manager was called
        self.mock_storage.load_data.assert_called_once_with("AAPL")
        self.mock_storage.save_data.assert_called_once()
    
    def test_preprocess_data_with_custom_steps(self):
        """Test preprocessing data with custom steps."""
        # Preprocess data with custom steps
        result = self.preprocessor.preprocess_data(["AAPL"], steps=["normalize"])
        
        # Verify the result
        self.assertIn("AAPL", result)
        self.assertTrue(result["AAPL"]["success"])
        self.assertEqual(result["AAPL"]["rows"], 5)
        self.assertEqual(len(result["AAPL"]["applied_steps"]), 1)
        self.assertEqual(result["AAPL"]["applied_steps"][0], "normalize")
        
        # Verify the storage manager was called
        self.mock_storage.load_data.assert_called_once_with("AAPL")
        self.mock_storage.save_data.assert_called_once()
    
    def test_preprocess_data_no_data(self):
        """Test preprocessing with no data."""
        # Configure mock to return empty DataFrame
        self.mock_storage.load_data.return_value = pd.DataFrame()
        
        # Preprocess data
        result = self.preprocessor.preprocess_data(["AAPL"])
        
        # Verify the result
        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["success"])
        self.assertEqual(result["AAPL"]["error"], "No data found")
        
        # Verify the storage manager was called
        self.mock_storage.load_data.assert_called_once_with("AAPL")
        self.mock_storage.save_data.assert_not_called()
    
    def test_preprocess_data_multiple_symbols(self):
        """Test preprocessing multiple symbols."""
        # Configure mock to return different data for different symbols
        self.mock_storage.load_data.side_effect = lambda symbol: {
            "AAPL": self.test_data,
            "MSFT": self.test_data,
            "INVALID": pd.DataFrame()
        }.get(symbol, pd.DataFrame())
        
        # Preprocess data
        result = self.preprocessor.preprocess_data(["AAPL", "MSFT", "INVALID"])
        
        # Verify the result
        self.assertEqual(len(result), 3)
        self.assertTrue(result["AAPL"]["success"])
        self.assertTrue(result["MSFT"]["success"])
        self.assertFalse(result["INVALID"]["success"])
        
        # Verify the storage manager was called
        self.assertEqual(self.mock_storage.load_data.call_count, 3)
        self.assertEqual(self.mock_storage.save_data.call_count, 2)
    
    def test_get_available_steps(self):
        """Test getting available preprocessing steps."""
        # Get available steps
        steps = self.preprocessor.get_available_steps()
        
        # Verify the steps
        self.assertEqual(len(steps), 4)
        self.assertIn("remove_outliers", steps)
        self.assertIn("fill_missing_values", steps)
        self.assertIn("normalize", steps)
        self.assertIn("calculate_indicators", steps)


if __name__ == '__main__':
    unittest.main()
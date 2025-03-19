"""
Tests for the LEAN implementation.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Import LEAN adapter if available
try:
    from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


@unittest.skipIf(not LEAN_AVAILABLE, "LEAN framework not available")
class TestLeanAdapter(unittest.TestCase):
    """Test cases for the LEAN adapter."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = None
        self.data_folder = os.path.join(self.temp_dir.name, "data")
        self.results_folder = os.path.join(self.temp_dir.name, "results")
        
        # Create folders
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    @patch("src.market_system.lean.lean_adapter.Engine")
    @patch("src.market_system.lean.lean_adapter.BacktestSettings")
    def test_initialization(self, mock_settings, mock_engine):
        """Test adapter initialization."""
        adapter = LeanAdapter(
            mode=BacktestMode.LEGACY,
            data_folder=self.data_folder,
            results_folder=self.results_folder
        )
        
        # Check if configuration is loaded correctly
        self.assertIsNotNone(adapter.config)
        self.assertEqual(adapter.mode, BacktestMode.LEGACY)
        self.assertEqual(adapter.data_folder, self.data_folder)
        self.assertEqual(adapter.results_folder, self.results_folder)
    
    @patch("src.market_system.lean.lean_adapter.importlib.import_module")
    def test_discover_algorithms(self, mock_import):
        """Test discovering LEAN algorithms."""
        # Mock the import result
        mock_module = MagicMock()
        mock_module.__path__ = ["/path/to/algorithms"]
        mock_import.return_value = mock_module
        
        # Create adapter
        adapter = LeanAdapter(
            mode=BacktestMode.LEGACY,
            data_folder=self.data_folder,
            results_folder=self.results_folder
        )
        
        # Check if algorithms are discovered
        self.assertIsInstance(adapter.algorithms, dict)
    
    @patch("src.market_system.lean.lean_adapter.Engine")
    @patch("src.market_system.lean.lean_adapter.BacktestSettings")
    def test_get_available_algorithms(self, mock_settings, mock_engine):
        """Test getting available LEAN algorithms."""
        adapter = LeanAdapter(
            mode=BacktestMode.LEGACY,
            data_folder=self.data_folder,
            results_folder=self.results_folder
        )
        
        # Mock algorithms
        adapter.algorithms = {
            "TestAlgorithm": MagicMock(),
            "AnotherAlgorithm": MagicMock()
        }
        
        # Get available algorithms
        algorithms = adapter.get_available_algorithms()
        
        # Check if algorithms are returned correctly
        self.assertEqual(len(algorithms), 2)
        self.assertIn("TestAlgorithm", algorithms)
        self.assertIn("AnotherAlgorithm", algorithms)
    
    @patch("src.market_system.lean.lean_adapter.pd.read_csv")
    @patch("src.market_system.lean.lean_adapter.pd.DataFrame.to_csv")
    def test_convert_data_to_lean(self, mock_to_csv, mock_read_csv):
        """Test converting data to LEAN format."""
        # Mock read_csv result
        mock_df = MagicMock()
        mock_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        mock_read_csv.return_value = mock_df
        
        # Create adapter
        adapter = LeanAdapter(
            mode=BacktestMode.LEGACY,
            data_folder=self.data_folder,
            results_folder=self.results_folder
        )
        
        # Convert data
        result = adapter.convert_data_to_lean(
            data_path="data/test.csv",
            symbol="TEST",
            output_path=os.path.join(self.data_folder, "test_output.csv")
        )
        
        # Check if data is converted correctly
        self.assertEqual(result, os.path.join(self.data_folder, "test_output.csv"))
        mock_to_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main() 
import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import tempfile
import json

from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode


class TestLeanAdapter(unittest.TestCase):
    """Test cases for the LEAN adapter."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary test directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.test_dir / "data"
        self.test_results_dir = self.test_dir / "results"
        
        self.test_data_dir.mkdir(exist_ok=True)
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Create sample config
        self.config_path = self.test_dir / "config.json"
        self.create_sample_config()
        
        # Create sample data
        self.create_sample_data()
        
        # Initialize LEAN adapter
        try:
            self.adapter = LeanAdapter(
                config_path=str(self.config_path),
                mode=BacktestMode.LEAN,
                data_folder=str(self.test_data_dir),
                results_folder=str(self.test_results_dir)
            )
            self.lean_available = True
        except ImportError:
            self.lean_available = False

    def tearDown(self):
        """Clean up after each test."""
        # Remove test directory and all contents
        shutil.rmtree(self.test_dir)

    def create_sample_config(self):
        """Create sample configuration file."""
        config = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "organization_id": "test_org",
            "project_id": "test_project",
            "default_cash": 100000,
            "default_market": "usa",
            "default_resolution": "daily",
            "default_algorithm": "TrendFollowingAlgorithm",
            "timeframe": "daily",
            "cache_data": True,
            "use_benchmark": True,
            "benchmark_symbol": "SPY",
            "slippage_model": "percentage",
            "slippage_value": 0.001,
            "commission_model": "fixed",
            "commission_value": 1.0
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        # Create LEAN data directory structure
        lean_data_dir = self.test_data_dir / "lean" / "equity" / "usa" / "daily"
        lean_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 30 days of sample data
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=i) for i in range(31)]
        
        # Create sample data with some price movements
        data = []
        base_price = 100.0
        
        for i, date in enumerate(dates):
            # Add some randomness and trend to price
            price_change = np.random.normal(0, 1) + 0.1  # Slight upward bias
            base_price *= (1 + price_change / 100)
            
            # Create OHLCV data
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = base_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = base_price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.normal(1000000, 200000))
            
            # Ensure high is highest and low is lowest
            high_price = max(open_price, high_price, close_price)
            low_price = min(open_price, low_price, close_price)
            
            data.append({
                'time': date.strftime('%Y-%m-%d'),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame and save to CSV in LEAN format
        df = pd.DataFrame(data)
        df.to_csv(lean_data_dir / "AAPL.csv", index=False)

    def test_initialization(self):
        """Test that the adapter initializes correctly."""
        if not self.lean_available:
            self.skipTest("LEAN framework not available")
            
        # Check adapter is initialized
        self.assertIsNotNone(self.adapter)
        self.assertEqual(self.adapter.mode, BacktestMode.LEAN)
        self.assertEqual(self.adapter.data_folder, str(self.test_data_dir))
        self.assertEqual(self.adapter.results_folder, str(self.test_results_dir))
        
        # Check config loaded correctly
        self.assertEqual(self.adapter.config['default_cash'], 100000)
        self.assertEqual(self.adapter.config['default_market'], 'usa')
        self.assertEqual(self.adapter.config['default_algorithm'], 'TrendFollowingAlgorithm')

    def test_available_algorithms(self):
        """Test that algorithm discovery works."""
        if not self.lean_available:
            self.skipTest("LEAN framework not available")
            
        # Get available algorithms
        algorithms = self.adapter.get_available_algorithms()
        
        # Verify we get a list (even if empty for the test environment)
        self.assertIsInstance(algorithms, list)

    def test_data_conversion(self):
        """Test data conversion to LEAN format."""
        if not self.lean_available:
            self.skipTest("LEAN framework not available")
            
        # Create a sample CSV file in non-LEAN format
        sample_file = self.test_dir / "sample_data.csv"
        
        # Generate sample data
        dates = [datetime.now() - timedelta(days=i) for i in range(10)]
        data = []
        
        for date in dates:
            data.append({
                'timestamp': date.strftime('%Y-%m-%d'),
                'open': 100 + np.random.random() * 5,
                'high': 105 + np.random.random() * 5,
                'low': 95 + np.random.random() * 5,
                'close': 100 + np.random.random() * 5,
                'volume': int(1000000 + np.random.random() * 500000)
            })
        
        # Save sample data
        pd.DataFrame(data).to_csv(sample_file, index=False)
        
        # Convert data
        output_path = self.adapter.convert_data_to_lean(
            data_path=str(sample_file),
            symbol="TEST"
        )
        
        # Check output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check output has correct format
        df = pd.read_csv(output_path)
        self.assertIn('time', df.columns)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)


if __name__ == '__main__':
    unittest.main() 
"""Unit tests for strategy analyzer module."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json
import os

from src.market_system.strategy_analyzer import StrategyAnalyzer, safe_float, safe_series_value

class TestStrategyAnalyzer(unittest.TestCase):
    """Test cases for StrategyAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data directory and sample market data."""
        # Create temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        cls.market_dir = Path(cls.test_dir) / 'test_market'
        cls.symbol_dir = cls.market_dir / 'TEST.SYM'
        cls.symbol_dir.mkdir(parents=True)
        
        # Create sample market data
        cls.sample_data = {
            'date': '2025-03-15',
            'symbol': 'TEST.SYM',
            'market': 'test_market',
            'close': 100.0,
            'volume': 10000,
            'trading_status': 'active',
            'currency': 'USD',
            'metrics': {
                'indicators': {
                    'vwap': 99.5,
                    'turnover': 995000
                },
                'returns': {
                    'daily': 0.01,
                    'intraday_range': 0.02
                }
            }
        }
        
        # Write sample data file
        data_file = cls.symbol_dir / '2025-03-15.json'
        with open(data_file, 'w') as f:
            json.dump(cls.sample_data, f)
    
    def setUp(self):
        """Initialize StrategyAnalyzer instance for each test."""
        self.analyzer = StrategyAnalyzer(self.test_dir)
    
    def test_safe_float_conversion(self):
        """Test safe float conversion utility."""
        # Test series conversion
        series = pd.Series([1.0, 2.0, 3.0])
        self.assertEqual(safe_float(series), 2.0)  # Should return mean
        
        # Test empty series
        empty_series = pd.Series([])
        self.assertEqual(safe_float(empty_series), 0.0)  # Should return default
        
        # Test single value series
        single_value = pd.Series([5.0])
        self.assertEqual(safe_float(single_value), 5.0)
        
        # Test numpy values
        self.assertEqual(safe_float(np.float32(1.5)), 1.5)
        
        # Test invalid values
        self.assertEqual(safe_float('invalid'), 0.0)
        self.assertEqual(safe_float(None), 0.0)
    
    def test_safe_series_value(self):
        """Test safe series value extraction."""
        # Test various input types
        series = pd.Series([1.0, 2.0, 3.0])
        self.assertEqual(safe_series_value(series), 2.0)  # Should return mean
        
        # Test numpy array
        array = np.array([1.0, 2.0, 3.0])
        self.assertEqual(safe_series_value(array), 2.0)
        
        # Test scalar values
        self.assertEqual(safe_series_value(5.0), 5.0)
        self.assertEqual(safe_series_value(np.float32(1.5)), 1.5)
        
        # Test invalid inputs
        self.assertEqual(safe_series_value('invalid'), 0.0)
        self.assertEqual(safe_series_value(None), 0.0)
    
    def test_calculate_indicators(self):
        """Test technical indicator calculations."""
        # Create test dataframe
        dates = pd.date_range('2025-01-01', '2025-03-15')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.normal(100, 10, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'vwap': np.random.normal(100, 5, len(dates)),
            'turnover': np.random.randint(100000, 1000000, len(dates)),
            'trading_status': 'active',
            'symbol': 'TEST.SYM'
        })
        
        # Calculate indicators
        result = self.analyzer._calculate_indicators(df)
        
        # Verify required columns exist
        required_indicators = ['SMA_20', 'SMA_50', 'volatility', 'volume_ma', 'volume_ratio']
        for indicator in required_indicators:
            self.assertIn(indicator, result.columns)
        
        # Verify calculations are reasonable
        self.assertTrue((result['SMA_20'] > 0).all())
        self.assertTrue((result['volatility'] >= 0).all())
        self.assertTrue((result['volume_ratio'] >= 0).all())
    
    def test_apply_strategy_filters(self):
        """Test strategy filter application."""
        # Create test dataframe with known signal conditions
        dates = pd.date_range('2025-01-01', '2025-03-15')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'vwap': 100 + np.cumsum(np.random.normal(0, 0.5, len(dates))),
            'turnover': np.random.randint(100000, 1000000, len(dates)),
            'trading_status': 'active',
            'symbol': 'TEST.SYM'
        })
        
        # Apply strategy filters
        results = self.analyzer.apply_strategy_filters(df)
        
        # Verify all strategy results exist
        self.assertIn('moving_average_crossover', results)
        self.assertIn('volatility_breakout', results)
        self.assertIn('volume_spike', results)
        
        # Verify each strategy returns a DataFrame
        for strategy_name, strategy_df in results.items():
            self.assertIsInstance(strategy_df, pd.DataFrame)
    
    def test_calculate_strategy_metrics(self):
        """Test strategy metrics calculation."""
        # Create test dataframe
        dates = pd.date_range('2025-01-01', '2025-03-15')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'vwap': 100 + np.cumsum(np.random.normal(0, 0.5, len(dates))),
            'turnover': np.random.randint(100000, 1000000, len(dates)),
            'trading_status': 'active',
            'symbol': 'TEST.SYM'
        })
        
        # Calculate metrics
        metrics = self.analyzer.calculate_strategy_metrics(df)
        
        # Verify required sections exist
        self.assertIn('strategies', metrics)
        self.assertIn('data_quality', metrics)
        self.assertIn('analysis_period', metrics)
        
        # Verify strategy metrics
        for strategy in metrics['strategies'].values():
            self.assertIn('status', strategy)
            if strategy['status'] == 'Success':
                self.assertIn('performance', strategy)
                self.assertIn('stats', strategy)
                self.assertIn('risk_metrics', strategy)
    
    def test_generate_report(self):
        """Test report generation with sample data."""
        # Generate report
        report = self.analyzer.generate_report('test_market')
        
        # Verify report structure
        self.assertIn('report_metadata', report)
        self.assertIn('market_summary', report)
        self.assertIn('strategies', report)
        self.assertIn('execution_metrics', report)
        
        # Verify report metadata
        metadata = report['report_metadata']
        self.assertEqual(metadata['report_info']['market'], 'test_market')
        self.assertIn('version', metadata['report_info'])
        
        # Verify execution metrics
        metrics = report['execution_metrics']
        self.assertIn('timing', metrics)
        self.assertIn('resources', metrics)
        self.assertIn('data_processed', metrics)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()
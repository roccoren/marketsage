"""
Tests for the strategy analyzer module.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .strategy_analyzer import StrategyAnalyzer

class TestStrategyAnalyzer(unittest.TestCase):
    """Test cases for StrategyAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_path = Path("test_data")
        self.analyzer = StrategyAnalyzer(self.test_data_path)
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'price': [100 + i + np.sin(i/5)*10 for i in range(len(dates))],
            'volume': [1000 + np.random.randint(-200, 200) for _ in range(len(dates))]
        })
    
    def test_moving_averages_calculation(self):
        """Test moving averages calculation."""
        df = self.analyzer._calculate_moving_averages(self.sample_data)
        
        self.assertTrue('SMA_20' in df.columns)
        self.assertTrue('SMA_50' in df.columns)
        
        # First 19 values should be NaN for SMA_20
        self.assertEqual(df['SMA_20'].isna().sum(), 19)
        
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        df = self.analyzer._calculate_volatility(self.sample_data)
        
        self.assertTrue('volatility' in df.columns)
        self.assertTrue(df['volatility'].dtype == float)
        
    def test_volume_trends_calculation(self):
        """Test volume trends calculation."""
        df = self.analyzer._calculate_volume_trends(self.sample_data)
        
        self.assertTrue('volume_ma' in df.columns)
        self.assertTrue('volume_ratio' in df.columns)
        self.assertTrue(all(df['volume_ratio'] >= 0))
        
    def test_strategy_filters(self):
        """Test strategy filters application."""
        # Prepare test data with technical indicators
        df = self.sample_data.copy()
        strategy_results = self.analyzer.apply_strategy_filters(df)
        
        self.assertIsInstance(strategy_results, dict)
        self.assertTrue('moving_average_crossover' in strategy_results)
        self.assertTrue('volatility_breakout' in strategy_results)
        self.assertTrue('volume_spike' in strategy_results)
        
        # Each strategy DataFrame should have the same columns as input plus indicators
        for strategy_df in strategy_results.values():
            self.assertTrue(isinstance(strategy_df, pd.DataFrame))
            self.assertTrue(all(col in strategy_df.columns for col in df.columns))
            
    def test_strategy_metrics_calculation(self):
        """Test strategy metrics calculation."""
        metrics = self.analyzer.calculate_strategy_metrics(self.sample_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertTrue('analysis_period' in metrics)
        self.assertTrue('total_stocks_analyzed' in metrics)
        self.assertTrue('strategies' in metrics)
        
        # Check metrics structure for each strategy
        for strategy_name, strategy_metrics in metrics['strategies'].items():
            self.assertTrue('matches' in strategy_metrics)
            self.assertTrue('match_rate' in strategy_metrics)
            self.assertTrue('success_rate' in strategy_metrics)
            self.assertTrue('average_return' in strategy_metrics)
            self.assertTrue('stats' in strategy_metrics)
            
            # Verify metric values are within expected ranges
            self.assertGreaterEqual(strategy_metrics['match_rate'], 0)
            self.assertLessEqual(strategy_metrics['match_rate'], 1)
            self.assertGreaterEqual(strategy_metrics['success_rate'], 0)
            self.assertLessEqual(strategy_metrics['success_rate'], 1)
            
    def test_report_generation(self):
        """Test comprehensive report generation."""
        # Mock the process_market_data method to return our sample data
        self.analyzer.processor.process_market_data = lambda x: self.sample_data
        
        report = self.analyzer.generate_report("test_market")
        
        # Verify report structure
        self.assertTrue('market_summary' in report)
        self.assertTrue('strategies' in report)
        self.assertTrue('report_generated' in report)
        
        # Verify market summary metrics
        market_summary = report['market_summary']
        self.assertTrue('total_trading_days' in market_summary)
        self.assertTrue('average_daily_volume' in market_summary)
        self.assertTrue('price_range' in market_summary)
        
        # Verify report timestamp
        self.assertTrue(datetime.fromisoformat(report['report_generated']))

if __name__ == '__main__':
    unittest.main()
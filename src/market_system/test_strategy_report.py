"""
Test script for strategy report generation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from generate_strategy_report import get_recent_strategy_signals, format_recent_signals_report

class TestStrategyReport(unittest.TestCase):
    def setUp(self):
        """Create sample market data with various test scenarios."""
        dates = pd.date_range(end=datetime.now(), periods=15)
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create test DataFrame with multiple scenarios
        data = []
        for date in dates:
            for symbol in symbols:
                # Regular data
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'price': np.random.uniform(100, 200),
                    'volume': np.random.uniform(1000, 5000),
                    'vwap': np.random.uniform(100, 200),
                    'turnover': np.random.uniform(100000, 500000),
                    'trading_status': 'active'
                })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators with proper error handling
        def safe_calculate_indicators(df):
            try:
                # Calculate basic indicators
                df['SMA_20'] = df.groupby('symbol')['price'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                df['SMA_50'] = df.groupby('symbol')['price'].transform(
                    lambda x: x.rolling(window=10, min_periods=1).mean()
                )
                df['volatility'] = df.groupby('symbol')['price'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std()
                )
                
                # Calculate volume metrics safely
                volume_ma = df.groupby('symbol')['volume'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                df['volume_ratio'] = (df['volume'] / volume_ma).replace([np.inf, -np.inf], np.nan)
                df['volume_ratio'] = df['volume_ratio'].fillna(1.0)  # Neutral value for missing data
                
                # Calculate VWAP ratio safely
                df['price_to_vwap'] = (df['price'] / df['vwap']).replace([np.inf, -np.inf], 1.0)
                
                return df
            except Exception as e:
                print(f"Error in indicator calculation: {str(e)}")
                return None
        
        processed_df = safe_calculate_indicators(df)
        if processed_df is None:
            raise ValueError("Failed to calculate indicators in test setup")
            
        self.test_data = processed_df
        
    def test_handle_missing_volume_ratio(self):
        """Test handling of missing volume_ratio column"""
        # Create data without volume_ratio
        df_no_volume = self.test_data.drop('volume_ratio', axis=1)
        
        # Should not raise KeyError
        signals = get_recent_strategy_signals(df_no_volume, days=10)
        self.assertIsInstance(signals, dict)
        self.assertTrue(all(isinstance(v, list) for v in signals.values()))

    def test_get_recent_strategy_signals(self):
        # Test signal detection
        signals = get_recent_strategy_signals(self.test_data, days=10)
        
        # Verify structure and content
        self.assertIsInstance(signals, dict)
        for strategy in ['moving_average_crossover', 'volatility_breakout', 'volume_spike']:
            self.assertIn(strategy, signals)
            if signals[strategy]:  # If signals were detected
                for stock in signals[strategy]:
                    # Verify required fields
                    self.assertIn('symbol', stock)
                    self.assertIn('trigger_dates', stock)
                    self.assertIn('signal_strength', stock)
                    self.assertIn('average_volume_ratio', stock)
                    self.assertIn('price_to_vwap', stock)
                    
                    # Verify data types
                    self.assertIsInstance(stock['symbol'], str)
                    self.assertIsInstance(stock['trigger_dates'], list)
                    self.assertIsInstance(stock['signal_strength'], float)
                    self.assertIsInstance(stock['average_volume_ratio'], float)
                    self.assertIsInstance(stock['price_to_vwap'], float)

    def test_format_recent_signals_report(self):
        # Test report formatting
        signals = get_recent_strategy_signals(self.test_data, days=10)
        report = format_recent_signals_report(signals)
        
        # Verify report structure
        self.assertIsInstance(report, str)
        self.assertIn("Recent Strategy Signals Analysis Report", report)
        self.assertIn("Last 10 Trading Days", report)
        
        # Verify signal content is included
        if any(signals.values()):
            for strategy_name in signals.keys():
                if signals[strategy_name]:
                    self.assertIn(strategy_name.replace('_', ' ').title(), report)

if __name__ == '__main__':
    unittest.main()
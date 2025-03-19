import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from strategies.backtester import Backtester
from strategies.macd import MACDStrategy
from utils.exchange_data_loader import ExchangeDataLoader


class TestMACDBacktesting(unittest.TestCase):
    def setUp(self):
        self.data_loader = ExchangeDataLoader()
        self.strategy = MACDStrategy()
        self.backtester = Backtester(self.strategy)

    def test_macd_calculation(self):
        """Test MACD calculation with sample data."""
        # Load real data from ASX for testing
        data = self.data_loader.load_stock_data("ASX", "A2M")
        self.assertIsNotNone(data, "Failed to load test data")

        # Calculate MACD
        macd_data = self.strategy.calculate_macd(data)

        self.assertIsInstance(macd_data, pd.DataFrame)
        self.assertIn("MACD", macd_data.columns)
        self.assertIn("Signal", macd_data.columns)
        self.assertIn("Histogram", macd_data.columns)

        # Verify MACD calculation logic
        self.assertEqual(len(macd_data), len(data))
        self.assertTrue(all(~np.isnan(macd_data["MACD"])))
        self.assertTrue(all(~np.isnan(macd_data["Signal"])))

    def test_backtest_execution(self):
        """Test full backtest execution."""
        data = self.data_loader.load_stock_data("ASX", "A2M")
        self.assertIsNotNone(data, "Failed to load test data")

        # Run backtest
        result = self.backtester.run(data)

        # Verify results structure
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "trades"))
        self.assertTrue(hasattr(result, "metrics"))

        # Verify metrics calculation
        metrics = result.metrics
        required_metrics = [
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "total_pnl",
            "avg_return",
            "max_drawdown",
            "sharpe_ratio",
            "profit_factor",
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsNotNone(metrics[metric])

    def test_cross_exchange_data(self):
        """Test MACD strategy across different exchanges."""
        test_cases = [
            ("ASX", "A2M"),
            ("NYSE", "AAPL"),
            ("NASDAQ", "MSFT"),
            ("HKEX", "0700"),
        ]

        for exchange, symbol in test_cases:
            with self.subTest(exchange=exchange, symbol=symbol):
                data = self.data_loader.load_stock_data(exchange, symbol)
                if data is not None:  # Only test if data exists
                    # Test MACD calculation
                    macd_data = self.strategy.calculate_macd(data)
                    self.assertIsInstance(macd_data, pd.DataFrame)

                    # Test signal generation
                    signals = self.strategy.analyze(data)
                    self.assertIsInstance(signals, list)

                    # Test backtest execution
                    result = self.backtester.run(data)
                    self.assertIsNotNone(result)

    def test_error_handling(self):
        """Test error handling for invalid data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        signals = self.strategy.analyze(empty_df)
        self.assertEqual(len(signals), 0)

        # Test with insufficient data
        small_df = pd.DataFrame(
            {"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        signals = self.strategy.analyze(small_df)
        self.assertEqual(len(signals), 0)

        # Test with missing required columns
        invalid_df = pd.DataFrame(
            {"Price": [100, 101, 102], "Volume": [1000, 1100, 1200]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        with self.assertRaises(Exception):
            self.strategy.analyze(invalid_df)

    def test_signal_generation(self):
        """Test MACD signal generation logic."""
        data = self.data_loader.load_stock_data("ASX", "A2M")
        self.assertIsNotNone(data, "Failed to load test data")

        # Get signals
        signals = self.strategy.analyze(data)

        # Verify signal structure
        for signal in signals:
            self.assertIn("Type", signal)
            self.assertIn(signal["Type"], ["BUY", "SELL"])
            self.assertIn("Price", signal)
            self.assertIn("SignalStrength", signal)
            self.assertGreaterEqual(signal["SignalStrength"], 0)


if __name__ == "__main__":
    unittest.main()

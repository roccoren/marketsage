import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategies.macd import MACDStrategy, SignalType


class TestMACDStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.strategy = MACDStrategy()

        # Create sample price data
        dates = pd.date_range(start="2023-01-01", end="2023-01-30", freq="D")
        self.sample_data = pd.DataFrame(
            {
                "Close": [100 + i + np.sin(i / 5) * 10 for i in range(len(dates))],
                "Volume": [
                    10000 + i * 100 + np.random.randint(-1000, 1000)
                    for i in range(len(dates))
                ],
            },
            index=dates,
        )

    def test_calculate_signal_strength(self):
        """Test signal strength calculation with various inputs."""
        # Test normal case
        strength = self.strategy.calculate_signal_strength(
            macd_hist=0.5, volume_ratio=1.5, price_change=0.03
        )
        self.assertTrue(0 <= strength <= 1)

        # Test extreme values
        max_strength = self.strategy.calculate_signal_strength(
            macd_hist=4.0, volume_ratio=3.0, price_change=0.2
        )
        self.assertAlmostEqual(max_strength, 1.0)

        # Test zero values
        min_strength = self.strategy.calculate_signal_strength(
            macd_hist=0.0, volume_ratio=1.0, price_change=0.0
        )
        self.assertAlmostEqual(min_strength, 0.0)

    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_data = self.strategy.calculate_macd(self.sample_data)

        # Check if all required components are present
        self.assertIn("MACD", macd_data.columns)
        self.assertIn("Signal", macd_data.columns)
        self.assertIn("Histogram", macd_data.columns)

        # Verify no NaN values
        self.assertFalse(macd_data["MACD"].isna().any())
        self.assertFalse(macd_data["Signal"].isna().any())
        self.assertFalse(macd_data["Histogram"].isna().any())

        # Verify histogram calculation
        np.testing.assert_array_almost_equal(
            macd_data["Histogram"], 2 * (macd_data["MACD"] - macd_data["Signal"])
        )

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Test with sample data
        rsi = self.strategy.calculate_rsi(self.sample_data)
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())

        # Test with stable then uptrend data
        uptrend_data = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0, 100.0, 100.1],  # Stable then small gain
                "Volume": [1000] * 5,
            },
            index=pd.date_range(start="2023-01-01", periods=5),
        )
        up_rsi = self.strategy.calculate_rsi(uptrend_data)

        print(f"\nUp RSI calculation:")
        delta = uptrend_data["Close"].diff()
        gains = (delta.where(delta > 0, 0)).fillna(0)
        losses = (-delta.where(delta < 0, 0)).fillna(0)
        print(f"Price changes: {delta.tail()}")
        print(f"Gains: {gains.tail()}")
        print(f"Losses: {losses.tail()}")
        print(f"RSI values: {up_rsi.tail()}")

        self.assertFalse(up_rsi.isna().any())
        self.assertTrue((up_rsi >= 0).all())
        self.assertTrue((up_rsi <= 100).all())

        # Test with downtrend data
        downtrend_data = pd.DataFrame(
            {
                "Close": [100.0, 99.9, 99.8, 99.7, 99.6],  # Consistent small losses
                "Volume": [1000] * 5,
            },
            index=pd.date_range(start="2023-01-01", periods=5),
        )
        down_rsi = self.strategy.calculate_rsi(downtrend_data)

        print(f"\nDown RSI calculation:")
        delta = downtrend_data["Close"].diff()
        gains = (delta.where(delta > 0, 0)).fillna(0)
        losses = (-delta.where(delta < 0, 0)).fillna(0)
        print(f"Price changes: {delta.tail()}")
        print(f"Gains: {gains.tail()}")
        print(f"Losses: {losses.tail()}")
        print(f"RSI values: {down_rsi.tail()}")

        self.assertFalse(down_rsi.isna().any())
        self.assertTrue((down_rsi >= 0).all())
        self.assertTrue((down_rsi <= 100).all())

    def test_detect_recent_golden_cross(self):
        """Test golden cross detection."""
        # Create data that causes a clear MACD crossover
        dates = pd.date_range(start="2023-01-01", periods=60)
        prices = []

        # Generate price pattern:
        # 1. Initial decline to get MACD below signal
        # 2. Brief consolidation to let MACD converge with signal
        # 3. Recovery phase to trigger crossover
        for i in range(len(dates)):
            if i < 25:
                # Initial downtrend
                prices.append(100 - i * 0.6)  # Steady decline
            elif i < 35:
                # Consolidation
                base = 85  # Flat price during consolidation
                prices.append(base)  # Flat price
            elif i < 45:
                # Initial recovery
                base = 85
                recovery = (i - 35) * 0.4  # Gentle recovery
                prices.append(base + recovery)
            else:
                base = 89
                prices.append(base + (i - 45) * 1.2)  # Sharp uptrend
                base = 94.5
                uptrend = (i - 45) * 0.8
                prices.append(base + uptrend)

        # Create DataFrame with test data
        cross_data = pd.DataFrame(
            {"Close": prices, "Volume": [10000] * len(dates)}, index=dates
        )

        # Calculate and print debug info
        macd_data = self.strategy.calculate_macd(cross_data)
        print("\nPrice pattern:")
        print(cross_data["Close"].tail(10))
        print("\nMACD values near potential cross:")
        print(macd_data.tail(10))
        print("\nMACD vs Signal differences:")
        print("(Negative means MACD is below signal line)")
        diffs = macd_data["MACD"] - macd_data["Signal"]
        print(diffs.tail(10))

        # Test golden cross detection
        result = self.strategy.detect_recent_golden_cross(cross_data)
        self.assertTrue(result, "Expected to detect a golden cross")

        # Test no golden cross in pure downtrend
        downtrend_data = pd.DataFrame(
            {"Close": [100 - i * 0.5 for i in range(60)], "Volume": [10000] * 60},
            index=dates,
        )
        self.assertFalse(self.strategy.detect_recent_golden_cross(downtrend_data))

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame(columns=["Close", "Volume"])
        self.assertEqual(len(self.strategy.analyze(empty_data)), 0)

        # Test with single data point
        single_data = pd.DataFrame(
            {"Close": [100], "Volume": [10000]}, index=[datetime.now()]
        )
        self.assertEqual(len(self.strategy.analyze(single_data)), 0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import pandas as pd

from utils.exchange_data_loader import ExchangeDataLoader


class TestExchangeDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = ExchangeDataLoader()

    def test_load_valid_asx_stock(self):
        """Test loading valid ASX stock data."""
        df = self.data_loader.load_stock_data("ASX", "A2M")

        # Check DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)

        # Check required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        self.assertTrue(all(df[col].dtype == float for col in required_columns[:-1]))
        self.assertTrue(df["Volume"].dtype in [int, float])

        # Check data is sorted
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_missing_file(self):
        """Test handling of missing file."""
        df = self.data_loader.load_stock_data("ASX", "INVALID")
        self.assertIsNone(df)

    def test_invalid_exchange(self):
        """Test handling of invalid exchange."""
        df = self.data_loader.load_stock_data("INVALID", "AAPL")
        self.assertIsNone(df)

    def test_load_stocks_across_exchanges(self):
        """Test loading stocks from different exchanges."""
        # Dictionary of exchange: stock symbol pairs that should exist
        test_stocks = {
            "ASX": "A2M",
            "NYSE": "AAPL",  # Assuming these stocks exist in your data
            "NASDAQ": "MSFT",
            "HKEX": "0700",
        }

        for exchange, symbol in test_stocks.items():
            with self.subTest(exchange=exchange, symbol=symbol):
                df = self.data_loader.load_stock_data(exchange, symbol)
                if df is not None:  # Only test if data exists
                    self.assertIsInstance(df, pd.DataFrame)
                    self.assertTrue(len(df) > 0)
                    self.assertIn("Close", df.columns)

    def test_data_continuity(self):
        """Test for data continuity and missing values."""
        df = self.data_loader.load_stock_data("ASX", "A2M")
        if df is not None:
            # Check for missing values
            self.assertEqual(df.isnull().sum().sum(), 0)

            # Check time series continuity (no huge gaps)
            date_diffs = df.index.to_series().diff().dropna()
            max_days_gap = date_diffs.max().days
            self.assertLess(
                max_days_gap, 10
            )  # Assuming no gaps > 10 days (adjust as needed)


if __name__ == "__main__":
    unittest.main()

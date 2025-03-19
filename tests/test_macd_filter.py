import json
import os
import unittest
from datetime import datetime, timedelta

import pandas as pd

from strategies.macd import MACDStrategy


class TestMACDFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample data for each exchange."""
        # Create test data directories
        cls.exchanges = ["NYSE", "NASDAQ", "ASX", "HKEX"]
        os.makedirs("data", exist_ok=True)
        for exchange in cls.exchanges:
            os.makedirs(f"data/{exchange}", exist_ok=True)

        # Create sample stock data
        cls._create_test_data()

    @classmethod
    def _create_test_data(cls):
        """Create sample stock data files for testing."""
        # Generate test data with different patterns
        patterns = {
            "uptrend": lambda i: 100 + i * 0.5,  # Steady uptrend
            "downtrend": lambda i: 100 - i * 0.3,  # Steady downtrend
            "volatile": lambda i: 100 + ((-1) ** i) * i * 0.4,  # Volatile pattern
            "sideways": lambda i: 100 + ((-1) ** i) * 0.2,  # Sideways movement
        }

        # Create one stock for each pattern in each exchange
        for exchange in cls.exchanges:
            for pattern_name, pattern_func in patterns.items():
                symbol = f"TEST_{pattern_name.upper()}"
                history = cls._generate_history(pattern_func)
                data = {"code": symbol, "history": history}

                filepath = f"data/{exchange}/{symbol}.json"
                with open(filepath, "w") as f:
                    json.dump(data, f)

    @classmethod
    def _generate_history(cls, pattern_func):
        """Generate historical price data using the given pattern function."""
        history = []
        base_date = datetime.now() - timedelta(days=100)

        for i in range(100):
            date = base_date + timedelta(days=i)
            price = pattern_func(i)
            history.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 100000 + ((-1) ** i) * 20000,
                }
            )

        return history

    def setUp(self):
        """Initialize strategy for each test."""
        self.strategy = MACDStrategy()

    def test_filter_stocks_returns_valid_structure(self):
        """Test that filter_stocks returns properly structured data."""
        # Run filter
        filtered_stocks = self.strategy.filter_stocks()

        # Basic structure tests
        self.assertIsInstance(filtered_stocks, list)
        if filtered_stocks:  # If any signals were generated
            first_signal = filtered_stocks[0]
            required_fields = [
                "Exchange",
                "Symbol",
                "Price",
                "Volume",
                "MACD",
                "Signal",
                "Histogram",
                "Type",
            ]
            for field in required_fields:
                self.assertIn(field, first_signal)

    def test_filter_stocks_processes_all_exchanges(self):
        """Test that all exchanges are processed."""
        # Get initial file counts
        exchange_file_counts = {
            exchange: len(os.listdir(f"data/{exchange}")) for exchange in self.exchanges
        }

        # Run filter
        _ = self.strategy.filter_stocks()

        # Check log file for processing confirmation
        with open("logs/macd_strategy.log", "r") as f:
            log_content = f.read()

        # Verify each exchange was processed
        for exchange, count in exchange_file_counts.items():
            self.assertIn(f"Processing {exchange} stocks", log_content)
            self.assertIn(f"Found {count} stock files in {exchange}", log_content)

    def test_filter_stocks_handles_invalid_data(self):
        """Test handling of invalid data files."""
        # Create invalid test file
        invalid_data = {"code": "INVALID", "history": []}  # Empty history

        filepath = "data/NYSE/INVALID.json"
        with open(filepath, "w") as f:
            json.dump(invalid_data, f)

        # Run filter
        filtered_stocks = self.strategy.filter_stocks()

        # Check logs for error handling
        with open("logs/macd_strategy.log", "r") as f:
            log_content = f.read()

        self.assertIn("Invalid/missing data for NYSE:INVALID", log_content)

    def test_filter_stocks_performance(self):
        """Test performance metrics are logged."""
        # Run filter
        _ = self.strategy.filter_stocks()

        # Check logs for performance metrics
        with open("logs/macd_strategy.log", "r") as f:
            log_content = f.read()

        self.assertIn("Performance - ", log_content)
        self.assertIn("Load:", log_content)
        self.assertIn("Total:", log_content)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Remove test data files
        for exchange in cls.exchanges:
            for pattern in ["UPTREND", "DOWNTREND", "VOLATILE", "SIDEWAYS", "INVALID"]:
                filepath = f"data/{exchange}/TEST_{pattern}.json"
                if os.path.exists(filepath):
                    os.remove(filepath)


if __name__ == "__main__":
    unittest.main()

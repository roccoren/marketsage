#!/usr/bin/env python3
"""
Unit tests for the SymbolManager module.
"""

import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import tempfile

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsage_datahub.symbols import SymbolManager


class TestSymbolManager(unittest.TestCase):
    """Test cases for the SymbolManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.symbols_file = os.path.join(self.temp_dir.name, "test_symbols.csv")
        
        # Create a test configuration
        self.config = {
            "symbols": {
                "local_file": self.symbols_file,
                "auto_update": False,
                "filters": {
                    "min_market_cap": 1000000000,
                    "exchanges": ["NASDAQ", "NYSE"]
                }
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_init_with_existing_file(self):
        """Test initialization with an existing symbols file."""
        # Create a symbols file
        with open(self.symbols_file, "w") as f:
            f.write("AAPL\nMSFT\nGOOGL\n")
        
        # Create the symbol manager
        manager = SymbolManager(config=self.config)
        
        # Verify the symbols were loaded
        self.assertEqual(len(manager.symbols), 3)
        self.assertIn("AAPL", manager.symbols)
        self.assertIn("MSFT", manager.symbols)
        self.assertIn("GOOGL", manager.symbols)
    
    def test_init_with_nonexistent_file(self):
        """Test initialization with a nonexistent symbols file."""
        # Ensure the file doesn't exist
        if os.path.exists(self.symbols_file):
            os.remove(self.symbols_file)
        
        # Create the symbol manager
        manager = SymbolManager(config=self.config)
        
        # Verify default symbols were used
        self.assertEqual(len(manager.symbols), 5)
        self.assertIn("AAPL", manager.symbols)
        self.assertIn("MSFT", manager.symbols)
        
        # Verify the file was created
        self.assertTrue(os.path.exists(self.symbols_file))
    
    def test_add_symbols(self):
        """Test adding symbols."""
        # Create the symbol manager with an empty file
        with open(self.symbols_file, "w") as f:
            f.write("")
        
        manager = SymbolManager(config=self.config)
        
        # Add symbols
        added = manager.add_symbols(["AAPL", "MSFT", "GOOGL"])
        
        # Verify the symbols were added
        self.assertEqual(len(added), 3)
        self.assertEqual(len(manager.symbols), 3)
        self.assertIn("AAPL", manager.symbols)
        
        # Add duplicate symbols
        added = manager.add_symbols(["AAPL", "AMZN"])
        
        # Verify only new symbols were added
        self.assertEqual(len(added), 1)
        self.assertEqual(len(manager.symbols), 4)
        self.assertIn("AMZN", manager.symbols)
        
        # Verify the file was updated
        with open(self.symbols_file, "r") as f:
            content = f.read()
        
        self.assertIn("AAPL", content)
        self.assertIn("AMZN", content)
    
    def test_remove_symbols(self):
        """Test removing symbols."""
        # Create the symbol manager with some symbols
        with open(self.symbols_file, "w") as f:
            f.write("AAPL\nMSFT\nGOOGL\nAMZN\n")
        
        manager = SymbolManager(config=self.config)
        
        # Remove symbols
        removed = manager.remove_symbols(["AAPL", "MSFT"])
        
        # Verify the symbols were removed
        self.assertEqual(len(removed), 2)
        self.assertEqual(len(manager.symbols), 2)
        self.assertNotIn("AAPL", manager.symbols)
        self.assertNotIn("MSFT", manager.symbols)
        self.assertIn("GOOGL", manager.symbols)
        self.assertIn("AMZN", manager.symbols)
        
        # Remove nonexistent symbols
        removed = manager.remove_symbols(["INVALID"])
        
        # Verify no symbols were removed
        self.assertEqual(len(removed), 0)
        self.assertEqual(len(manager.symbols), 2)
        
        # Verify the file was updated
        with open(self.symbols_file, "r") as f:
            content = f.read()
        
        self.assertNotIn("AAPL", content)
        self.assertIn("GOOGL", content)
    
    def test_get_symbols_with_filters(self):
        """Test getting symbols with filters."""
        # Create the symbol manager
        manager = SymbolManager(config=self.config)
        
        # Mock the get_symbol_details method
        manager.get_symbol_details = MagicMock(side_effect=lambda symbol: {
            "AAPL": {"symbol": "AAPL", "exchange": "NASDAQ", "market_cap": 2000000000},
            "MSFT": {"symbol": "MSFT", "exchange": "NASDAQ", "market_cap": 1500000000},
            "GOOGL": {"symbol": "GOOGL", "exchange": "NASDAQ", "market_cap": 1800000000},
            "SMALL": {"symbol": "SMALL", "exchange": "NASDAQ", "market_cap": 500000000},
            "OTHER": {"symbol": "OTHER", "exchange": "OTHER", "market_cap": 2000000000},
        }.get(symbol, {}))
        
        # Set test symbols
        manager.symbols = ["AAPL", "MSFT", "GOOGL", "SMALL", "OTHER"]
        
        # Test with exchange filter
        filtered = manager.get_symbols({"exchanges": ["NASDAQ"]})
        self.assertEqual(len(filtered), 4)
        self.assertIn("AAPL", filtered)
        self.assertIn("MSFT", filtered)
        self.assertIn("GOOGL", filtered)
        self.assertIn("SMALL", filtered)
        self.assertNotIn("OTHER", filtered)
        
        # Test with market cap filter
        filtered = manager.get_symbols({"min_market_cap": 1700000000})
        self.assertEqual(len(filtered), 3)
        self.assertIn("AAPL", filtered)
        self.assertNotIn("MSFT", filtered)
        self.assertIn("GOOGL", filtered)
        self.assertNotIn("SMALL", filtered)
        self.assertIn("OTHER", filtered)
        
        # Test with combined filters
        filtered = manager.get_symbols({
            "exchanges": ["NASDAQ"],
            "min_market_cap": 1700000000
        })
        self.assertEqual(len(filtered), 2)
        self.assertIn("AAPL", filtered)
        self.assertNotIn("MSFT", filtered)
        self.assertIn("GOOGL", filtered)
        self.assertNotIn("SMALL", filtered)
        self.assertNotIn("OTHER", filtered)
    
    def test_search_symbols(self):
        """Test searching for symbols."""
        # Create the symbol manager
        manager = SymbolManager(config=self.config)
        
        # Mock the get_symbol_details method
        manager.get_symbol_details = MagicMock(side_effect=lambda symbol: {
            "AAPL": {"symbol": "AAPL", "name": "Apple Inc."},
            "MSFT": {"symbol": "MSFT", "name": "Microsoft Corporation"},
            "GOOGL": {"symbol": "GOOGL", "name": "Alphabet Inc."},
            "AMZN": {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            "META": {"symbol": "META", "name": "Meta Platforms, Inc."},
        }.get(symbol, {}))
        
        # Set test symbols
        manager.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        # Test search by symbol
        results = manager.search_symbols("AA")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["symbol"], "AAPL")
        self.assertEqual(results[1]["symbol"], "AMZN")
        
        # Test search by name
        results = manager.search_symbols("inc")
        self.assertEqual(len(results), 3)
        symbols = [r["symbol"] for r in results]
        self.assertIn("AAPL", symbols)
        self.assertIn("GOOGL", symbols)
        self.assertIn("AMZN", symbols)
        
        # Test with limit
        results = manager.search_symbols("inc", limit=2)
        self.assertEqual(len(results), 2)


if __name__ == '__main__':
    unittest.main()
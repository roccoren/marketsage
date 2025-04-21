#!/usr/bin/env python3
"""
Unit tests for the CLI module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsage_datahub.cli import (
    setup_logging,
    load_config,
    download_command,
    symbols_command,
    preprocess_command,
    main
)


class TestCLI(unittest.TestCase):
    """Test cases for the CLI module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_data_acquisition = MagicMock()
        self.mock_symbol_manager = MagicMock()
        self.mock_storage_manager = MagicMock()
        self.mock_preprocessor = MagicMock()
        
        # Create test configuration
        self.config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "acquisition": {
                "default_start_date": "2020-01-01"
            }
        }
    
    @patch('marketsage_datahub.cli.logging')
    def test_setup_logging(self, mock_logging):
        """Test setting up logging."""
        # Call the function
        setup_logging(self.config)
        
        # Verify logging was configured
        mock_logging.basicConfig.assert_called_once()
    
    @patch('marketsage_datahub.cli.os.path.exists')
    @patch('marketsage_datahub.cli.yaml.safe_load')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_load_config(self, mock_open, mock_safe_load, mock_exists):
        """Test loading configuration."""
        # Configure mocks
        mock_exists.return_value = True
        mock_safe_load.return_value = self.config
        
        # Call the function
        result = load_config("config.yaml")
        
        # Verify the result
        self.assertEqual(result, self.config)
        mock_open.assert_called_once_with("config.yaml", "r")
        mock_safe_load.assert_called_once()
    
    @patch('marketsage_datahub.cli.SymbolManager')
    @patch('marketsage_datahub.cli.StorageManager')
    @patch('marketsage_datahub.cli.DataAcquisition')
    def test_download_command(self, mock_data_acquisition_cls, mock_storage_manager_cls, mock_symbol_manager_cls):
        """Test download command."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        mock_storage_manager_cls.return_value = self.mock_storage_manager
        mock_data_acquisition_cls.return_value = self.mock_data_acquisition
        
        self.mock_symbol_manager.get_symbols.return_value = ["AAPL", "MSFT"]
        self.mock_data_acquisition.download_data.return_value = {
            "AAPL": {"success": True},
            "MSFT": {"success": True}
        }
        
        # Create args
        args = argparse.Namespace(
            symbols=None,
            start_date="2023-01-01",
            end_date=None
        )
        
        # Call the function
        download_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        mock_storage_manager_cls.assert_called_once_with(config=self.config)
        mock_data_acquisition_cls.assert_called_once_with(
            storage_manager=self.mock_storage_manager,
            config=self.config
        )
        
        self.mock_symbol_manager.get_symbols.assert_called_once()
        self.mock_data_acquisition.download_data.assert_called_once_with(
            ["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date=None
        )
    
    @patch('marketsage_datahub.cli.SymbolManager')
    def test_symbols_command_list(self, mock_symbol_manager_cls):
        """Test symbols command with list action."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        self.mock_symbol_manager.get_symbols.return_value = ["AAPL", "MSFT"]
        
        # Create args
        args = argparse.Namespace(
            action="list",
            symbols=None,
            query=None,
            limit=10
        )
        
        # Call the function
        symbols_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        self.mock_symbol_manager.get_symbols.assert_called_once()
    
    @patch('marketsage_datahub.cli.SymbolManager')
    def test_symbols_command_add(self, mock_symbol_manager_cls):
        """Test symbols command with add action."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        self.mock_symbol_manager.add_symbols.return_value = ["AAPL", "MSFT"]
        
        # Create args
        args = argparse.Namespace(
            action="add",
            symbols="AAPL,MSFT",
            query=None,
            limit=10
        )
        
        # Call the function
        symbols_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        self.mock_symbol_manager.add_symbols.assert_called_once_with(["AAPL", "MSFT"])
    
    @patch('marketsage_datahub.cli.SymbolManager')
    def test_symbols_command_remove(self, mock_symbol_manager_cls):
        """Test symbols command with remove action."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        self.mock_symbol_manager.remove_symbols.return_value = ["AAPL"]
        
        # Create args
        args = argparse.Namespace(
            action="remove",
            symbols="AAPL",
            query=None,
            limit=10
        )
        
        # Call the function
        symbols_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        self.mock_symbol_manager.remove_symbols.assert_called_once_with(["AAPL"])
    
    @patch('marketsage_datahub.cli.SymbolManager')
    def test_symbols_command_search(self, mock_symbol_manager_cls):
        """Test symbols command with search action."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        self.mock_symbol_manager.search_symbols.return_value = [
            {"symbol": "AAPL", "name": "Apple Inc."}
        ]
        
        # Create args
        args = argparse.Namespace(
            action="search",
            symbols=None,
            query="Apple",
            limit=10
        )
        
        # Call the function
        symbols_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        self.mock_symbol_manager.search_symbols.assert_called_once_with("Apple", limit=10)
    
    @patch('marketsage_datahub.cli.SymbolManager')
    @patch('marketsage_datahub.cli.StorageManager')
    @patch('marketsage_datahub.cli.DataPreprocessor')
    def test_preprocess_command(self, mock_preprocessor_cls, mock_storage_manager_cls, mock_symbol_manager_cls):
        """Test preprocess command."""
        # Configure mocks
        mock_symbol_manager_cls.return_value = self.mock_symbol_manager
        mock_storage_manager_cls.return_value = self.mock_storage_manager
        mock_preprocessor_cls.return_value = self.mock_preprocessor
        
        self.mock_symbol_manager.get_symbols.return_value = ["AAPL", "MSFT"]
        self.mock_preprocessor.preprocess_data.return_value = {
            "AAPL": {"success": True},
            "MSFT": {"success": True}
        }
        
        # Create args
        args = argparse.Namespace(
            symbols=None,
            steps="remove_outliers,fill_missing_values"
        )
        
        # Call the function
        preprocess_command(args, self.config)
        
        # Verify the calls
        mock_symbol_manager_cls.assert_called_once_with(config=self.config)
        mock_storage_manager_cls.assert_called_once_with(config=self.config)
        mock_preprocessor_cls.assert_called_once_with(
            storage_manager=self.mock_storage_manager,
            config=self.config
        )
        
        self.mock_symbol_manager.get_symbols.assert_called_once()
        self.mock_preprocessor.preprocess_data.assert_called_once_with(
            ["AAPL", "MSFT"],
            steps=["remove_outliers", "fill_missing_values"]
        )
    
    @patch('marketsage_datahub.cli.argparse.ArgumentParser')
    @patch('marketsage_datahub.cli.load_config')
    @patch('marketsage_datahub.cli.setup_logging')
    @patch('marketsage_datahub.cli.download_command')
    @patch('marketsage_datahub.cli.symbols_command')
    @patch('marketsage_datahub.cli.preprocess_command')
    def test_main(self, mock_preprocess, mock_symbols, mock_download, mock_setup_logging, mock_load_config, mock_parser_cls):
        """Test main function."""
        # Configure mocks
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = argparse.Namespace(
            config=None,
            debug=False,
            command="download"
        )
        mock_load_config.return_value = self.config
        
        # Call the function
        main()
        
        # Verify the calls
        mock_parser_cls.assert_called_once()
        mock_parser.parse_args.assert_called_once()
        mock_load_config.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_download.assert_called_once()
        mock_symbols.assert_not_called()
        mock_preprocess.assert_not_called()


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3
"""
MarketSage DataHub CLI

Command-line interface for the DataHub component of the MarketSage ecosystem.
Provides commands for data acquisition, symbol management, and storage operations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import DataHub modules
from marketsage_datahub.acquisition import DataAcquisition
from marketsage_datahub.symbols import SymbolManager
from marketsage_datahub.storage import StorageManager
from marketsage_datahub.preprocessing import DataPreprocessor


def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging based on configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file")
    
    handlers = []
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        # Try to find config in standard locations
        config_locations = [
            "config/config.yaml",
            os.path.expanduser("~/.marketsage/config.yaml"),
            "/etc/marketsage/config.yaml",
        ]
        
        for loc in config_locations:
            if os.path.exists(loc):
                config_path = loc
                break
    
    if not config_path or not os.path.exists(config_path):
        print(f"Config file not found. Using default configuration.")
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Handle the download command."""
    acquisition_config = config.get("acquisition", {})
    
    # Initialize components
    symbol_manager = SymbolManager(config=config)
    storage_manager = StorageManager(config=config)
    data_acquisition = DataAcquisition(storage_manager=storage_manager, config=config)
    
    # Get symbols to download
    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = symbol_manager.get_symbols()
    
    if not symbols:
        print("No symbols specified and no symbols found in the symbol manager.")
        return
    
    # Set date range
    start_date = args.start_date or acquisition_config.get("default_start_date", "2020-01-01")
    end_date = args.end_date
    
    # Download data
    print(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date or 'now'}...")
    result = data_acquisition.download_data(symbols, start_date=start_date, end_date=end_date)
    
    # Print summary
    success_count = sum(1 for status in result.values() if status.get("success"))
    print(f"Download completed: {success_count}/{len(symbols)} symbols successful.")


def symbols_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Handle the symbols command."""
    symbol_manager = SymbolManager(config=config)
    
    if args.action == "list":
        symbols = symbol_manager.get_symbols()
        print(f"Available symbols ({len(symbols)}):")
        for symbol in symbols:
            print(symbol)
    
    elif args.action == "add":
        if not args.symbols:
            print("Error: No symbols specified to add.")
            return
        
        symbols = args.symbols.split(",")
        added = symbol_manager.add_symbols(symbols)
        print(f"Added {len(added)} symbols.")
    
    elif args.action == "remove":
        if not args.symbols:
            print("Error: No symbols specified to remove.")
            return
        
        symbols = args.symbols.split(",")
        removed = symbol_manager.remove_symbols(symbols)
        print(f"Removed {len(removed)} symbols.")
    
    elif args.action == "search":
        if not args.query:
            print("Error: No search query provided.")
            return
        
        results = symbol_manager.search_symbols(args.query, limit=args.limit)
        print(f"Search results for '{args.query}' ({len(results)} matches):")
        for symbol_info in results:
            print(f"{symbol_info['symbol']}: {symbol_info.get('name', 'Unknown')}")


def preprocess_command(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Handle the preprocess command."""
    # Initialize components
    symbol_manager = SymbolManager(config=config)
    storage_manager = StorageManager(config=config)
    preprocessor = DataPreprocessor(storage_manager=storage_manager, config=config)
    
    # Get symbols to preprocess
    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = symbol_manager.get_symbols()
    
    if not symbols:
        print("No symbols specified and no symbols found in the symbol manager.")
        return
    
    # Set preprocessing steps
    steps = args.steps.split(",") if args.steps else None
    
    # Preprocess data
    print(f"Preprocessing data for {len(symbols)} symbols...")
    result = preprocessor.preprocess_data(symbols, steps=steps)
    
    # Print summary
    success_count = sum(1 for status in result.values() if status.get("success"))
    print(f"Preprocessing completed: {success_count}/{len(symbols)} symbols successful.")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="MarketSage DataHub CLI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download stock data")
    download_parser.add_argument("--symbols", help="Comma-separated list of symbols to download")
    download_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    
    # Symbols command
    symbols_parser = subparsers.add_parser("symbols", help="Manage stock symbols")
    symbols_parser.add_argument("action", choices=["list", "add", "remove", "search"], help="Action to perform")
    symbols_parser.add_argument("--symbols", help="Comma-separated list of symbols")
    symbols_parser.add_argument("--query", help="Search query for symbols")
    symbols_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess stock data")
    preprocess_parser.add_argument("--symbols", help="Comma-separated list of symbols to preprocess")
    preprocess_parser.add_argument("--steps", help="Comma-separated list of preprocessing steps")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    if args.debug:
        config.setdefault("logging", {})["level"] = "DEBUG"
    setup_logging(config)
    
    # Execute command
    if args.command == "download":
        download_command(args, config)
    elif args.command == "symbols":
        symbols_command(args, config)
    elif args.command == "preprocess":
        preprocess_command(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
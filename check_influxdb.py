#!/usr/bin/env python3
"""
Script to check if data has been successfully written to InfluxDB.
"""

import sys
from modules.storage import StorageManager
from modules.config_manager import get_config_manager
import pandas as pd

def main():
    # Initialize configuration manager with default config file
    config_manager = get_config_manager("config/default_config.yaml")
    print("Configuration loaded")
    
    # Create storage manager with InfluxDB backend
    storage_manager = StorageManager(backend_type="influxdb")
    print("Storage manager initialized with InfluxDB backend")
    
    # List all available symbols in InfluxDB
    backend = storage_manager.get_backend()
    symbols = backend.list_symbols()
    
    print(f"Found {len(symbols)} symbols in InfluxDB:")
    for symbol in symbols:
        print(f"- {symbol}")
    
    # If symbols are found, show data for the first symbol as an example
    if symbols:
        symbol = symbols[0]
        print(f"\nFetching data for {symbol}...")
        data = backend.load_data(symbol)
        
        if not data.empty:
            print(f"Successfully loaded {len(data)} data points for {symbol}")
            print("\nFirst 5 rows:")
            print(data.head())
            print("\nLast 5 rows:")
            print(data.tail())
        else:
            print(f"No data found for {symbol}")

if __name__ == "__main__":
    try:
        main()
        print("\nCheck completed successfully")
    except Exception as e:
        print(f"Check failed: {e}")
        sys.exit(1)
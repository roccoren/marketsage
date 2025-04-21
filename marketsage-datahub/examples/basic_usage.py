#!/usr/bin/env python3
"""
Basic usage example for the MarketSage DataHub component.

This example demonstrates how to:
1. Initialize the DataHub components
2. Download stock data
3. Preprocess the data
4. Access the processed data
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import DataHub components
from marketsage_datahub.acquisition import DataAcquisition
from marketsage_datahub.symbols import SymbolManager
from marketsage_datahub.storage import StorageManager
from marketsage_datahub.preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate DataHub usage."""
    # Create data directories if they don't exist
    os.makedirs("data/csv/stock_data", exist_ok=True)
    
    # Initialize components with default configuration
    symbol_manager = SymbolManager()
    storage_manager = StorageManager()
    data_acquisition = DataAcquisition(storage_manager=storage_manager)
    data_preprocessor = DataPreprocessor(storage_manager=storage_manager)
    
    # Define symbols to download
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Download data
    logger.info(f"Downloading data for {len(symbols)} symbols...")
    download_results = data_acquisition.download_data(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Print download results
    for symbol, result in download_results.items():
        if result.get("success"):
            logger.info(f"Successfully downloaded {result.get('rows')} rows for {symbol}")
        else:
            logger.error(f"Failed to download data for {symbol}: {result.get('error')}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocess_results = data_preprocessor.preprocess_data(
        symbols=symbols,
        steps=["remove_outliers", "fill_missing_values", "calculate_indicators"]
    )
    
    # Print preprocessing results
    for symbol, result in preprocess_results.items():
        if result.get("success"):
            logger.info(f"Successfully preprocessed {symbol} with steps: {result.get('applied_steps')}")
        else:
            logger.error(f"Failed to preprocess data for {symbol}: {result.get('error')}")
    
    # Load and display processed data
    for symbol in symbols:
        processed_data = storage_manager.load_data(f"{symbol}_processed")
        if not processed_data.empty:
            logger.info(f"Processed data for {symbol}:")
            logger.info(f"Shape: {processed_data.shape}")
            logger.info(f"Columns: {processed_data.columns.tolist()}")
            logger.info(f"First 5 rows:\n{processed_data.head()}")
        else:
            logger.warning(f"No processed data found for {symbol}")


if __name__ == "__main__":
    main()
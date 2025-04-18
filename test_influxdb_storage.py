#!/usr/bin/env python3
"""
Test script for the InfluxDB storage backend.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from modules.storage import StorageManager


def create_sample_data(symbol, days=30):
    """Create sample stock data for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random price data
    close_price = 100.0
    data = []
    for date in dates:
        # Random daily change between -2% and 2%
        daily_change = random.uniform(-0.02, 0.02)
        close_price = close_price * (1 + daily_change)
        
        # Generate OHLC data
        open_price = close_price * random.uniform(0.99, 1.01)
        high_price = max(open_price, close_price) * random.uniform(1.001, 1.02)
        low_price = min(open_price, close_price) * random.uniform(0.98, 0.999)
        volume = int(random.uniform(100000, 1000000))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


def test_stock_data_storage(storage_manager, symbol):
    """Test saving and loading stock data."""
    print(f"\n=== Testing stock data storage for {symbol} ===")
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(symbol)
    print(f"Sample data shape: {data.shape}")
    print(data.head())
    
    # Save data
    print("\nSaving data to InfluxDB...")
    success = storage_manager.save_data(symbol, data)
    print(f"Save success: {success}")
    
    # Load data
    print("\nLoading data from InfluxDB...")
    loaded_data = storage_manager.load_data(symbol)
    print(f"Loaded data shape: {loaded_data.shape}")
    print(loaded_data.head())
    
    # Verify data
    if loaded_data.empty:
        print("ERROR: Failed to load data")
        return False
    
    # Check if the data is similar (may not be exactly the same due to floating point precision)
    print("\nVerifying data...")
    date_match = set(data.index) == set(loaded_data.index)
    print(f"Date match: {date_match}")
    
    # Check if columns match
    columns_match = set(data.columns) == set(loaded_data.columns)
    print(f"Columns match: {columns_match}")
    
    # Check if values are close
    sample_date = data.index[0]
    if sample_date in loaded_data.index:
        original_row = data.loc[sample_date]
        loaded_row = loaded_data.loc[sample_date]
        print(f"\nSample comparison for {sample_date}:")
        print(f"Original: {original_row}")
        print(f"Loaded:   {loaded_row}")
    
    return True


def test_model_params_storage(storage_manager, symbol):
    """Test saving and loading model parameters with versioning."""
    print(f"\n=== Testing model parameters storage for {symbol} ===")
    
    model_name = "test_model"
    
    # Create sample model parameters
    params_v1 = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "layers": [64, 32, 16],
        "dropout": 0.2
    }
    
    # Save version 1
    print("\nSaving model parameters (version 1)...")
    success = storage_manager.save_model_params(
        symbol, 
        model_name, 
        params_v1, 
        version="v1", 
        description="Initial model",
        metrics={"accuracy": 0.85, "loss": 0.23}
    )
    print(f"Save success: {success}")
    
    # Create updated parameters
    params_v2 = {
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 200,
        "layers": [128, 64, 32],
        "dropout": 0.3
    }
    
    # Save version 2
    print("\nSaving model parameters (version 2)...")
    success = storage_manager.save_model_params(
        symbol, 
        model_name, 
        params_v2, 
        version="v2", 
        description="Improved model",
        metrics={"accuracy": 0.89, "loss": 0.18}
    )
    print(f"Save success: {success}")
    
    # List versions
    print("\nListing model versions...")
    if hasattr(storage_manager.backend, 'list_model_versions'):
        versions = storage_manager.backend.list_model_versions(symbol, model_name)
        for version in versions:
            print(f"Version: {version['version']}")
            print(f"  Status: {version['status']}")
            print(f"  Timestamp: {version['timestamp']}")
            print(f"  Description: {version['description']}")
            if 'accuracy' in version:
                print(f"  Accuracy: {version['accuracy']}")
            if 'loss' in version:
                print(f"  Loss: {version['loss']}")
    else:
        print("Backend doesn't support listing model versions")
    
    # Load active version (should be v2)
    print("\nLoading active model parameters...")
    loaded_params = storage_manager.load_model_params(symbol, model_name)
    print(f"Loaded parameters: {loaded_params}")
    
    # Test rollback
    if hasattr(storage_manager.backend, 'rollback_to_version'):
        print("\nRolling back to version 1...")
        success = storage_manager.backend.rollback_to_version(symbol, model_name, "v1")
        print(f"Rollback success: {success}")
        
        # List versions after rollback
        print("\nListing model versions after rollback...")
        versions = storage_manager.backend.list_model_versions(symbol, model_name)
        for version in versions:
            print(f"Version: {version['version']}")
            print(f"  Status: {version['status']}")
    else:
        print("Backend doesn't support rollback")
    
    # Load active version after rollback (should be v1_restored)
    print("\nLoading active model parameters after rollback...")
    loaded_params = storage_manager.load_model_params(symbol, model_name)
    print(f"Loaded parameters: {loaded_params}")
    
    return True


def test_training_history(storage_manager, symbol):
    """Test saving and loading training history."""
    print(f"\n=== Testing training history storage for {symbol} ===")
    
    model_name = "test_model"
    version = "v1"
    
    # Create sample training history
    epochs = 10
    history = {
        "loss": [0.5 - 0.03 * i for i in range(epochs)],
        "val_loss": [0.55 - 0.025 * i for i in range(epochs)],
        "accuracy": [0.7 + 0.02 * i for i in range(epochs)],
        "val_accuracy": [0.65 + 0.02 * i for i in range(epochs)]
    }
    
    # Save training history
    print("\nSaving training history...")
    if hasattr(storage_manager.backend, 'save_training_history'):
        success = storage_manager.backend.save_training_history(symbol, model_name, version, history)
        print(f"Save success: {success}")
        
        # Load training history
        print("\nLoading training history...")
        loaded_history = storage_manager.backend.get_training_history(symbol, model_name, version)
        print(f"Loaded history: {loaded_history}")
    else:
        print("Backend doesn't support training history")
    
    return True


def main():
    """Main function."""
    # Initialize storage manager with InfluxDB backend
    print("Initializing InfluxDB storage manager...")
    storage_manager = StorageManager(
        backend_type="influxdb",
        url=os.environ.get("INFLUXDB_URL", "http://localhost:8086"),
        token=os.environ.get("INFLUXDB_TOKEN", "stockscantoken"),
        org=os.environ.get("INFLUXDB_ORG", "stockscan"),
        bucket=os.environ.get("INFLUXDB_BUCKET", "stock_data")
    )
    
    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        test_stock_data_storage(storage_manager, symbol)
    
    # Test model parameters storage with versioning
    test_model_params_storage(storage_manager, "AAPL")
    
    # Test training history storage
    test_training_history(storage_manager, "AAPL")
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
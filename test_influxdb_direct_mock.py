#!/usr/bin/env python3
"""
Test script for the InfluxDB storage backend using a direct mock implementation.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import random
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from modules.storage_backends.mock_influxdb import MockInfluxDBClient, Point


class MockInfluxDBStorage:
    """Mock implementation of InfluxDB storage backend for testing."""
    
    def __init__(self, url: str = "mock://localhost:8086", 
                 token: str = "mock-token", 
                 org: str = "stockscan",
                 bucket: str = "stock_data"):
        """Initialize mock InfluxDB storage."""
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        
        # Initialize client
        self.client = MockInfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api()
        self.query_api = self.client.query_api()
        
        # Store data in memory
        self.stock_data = {}
        self.model_params = {}
        self.model_history = {}
        
        print(f"Initialized mock InfluxDB storage with bucket: {bucket}")
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save data for a symbol to mock storage."""
        try:
            # Store data in memory
            self.stock_data[symbol] = data.copy()
            
            # Create points for the write API
            points = []
            for timestamp, row in data.iterrows():
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .time(timestamp)
                
                # Add all numeric columns as fields
                for col, value in row.items():
                    if pd.api.types.is_numeric_dtype(type(value)):
                        point = point.field(col.lower(), float(value))
                
                points.append(point)
            
            # Write points to mock InfluxDB
            self.write_api.write(bucket=self.bucket, record=points)
            return True
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load data for a symbol from mock storage."""
        try:
            if symbol not in self.stock_data:
                return pd.DataFrame()
            
            df = self.stock_data[symbol].copy()
            
            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_model_params(self, symbol: str, model_name: str, params: dict, 
                         version: str = None, description: str = None, metrics: dict = None) -> bool:
        """Save model parameters to mock storage."""
        try:
            # Generate version ID if not provided
            if not version:
                version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create model params entry
            if symbol not in self.model_params:
                self.model_params[symbol] = {}
            
            if model_name not in self.model_params[symbol]:
                self.model_params[symbol][model_name] = {}
            
            # Mark previous active versions as superseded
            for ver, ver_data in self.model_params[symbol][model_name].items():
                if ver_data.get("status") == "active":
                    self.model_params[symbol][model_name][ver]["status"] = "superseded"
            
            # Store new version
            self.model_params[symbol][model_name][version] = {
                "params": params,
                "status": "active",
                "timestamp": datetime.now(),
                "description": description or "",
                "metrics": metrics or {}
            }
            
            # Create point for the write API
            params_json = json.dumps(params, default=str)
            point = Point("model_params") \
                .tag("symbol", symbol) \
                .tag("model_name", model_name) \
                .tag("version", version) \
                .tag("status", "active") \
                .field("params", params_json)
            
            if description:
                point = point.field("description", description)
                
            if metrics:
                metrics_json = json.dumps(metrics, default=str)
                point = point.field("metrics", metrics_json)
                
                # Add individual metrics as separate fields
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        point = point.field(f"metric_{key}", value)
            
            # Write point to mock InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            
            return True
        except Exception as e:
            print(f"Error saving model parameters for {symbol}/{model_name}: {e}")
            return False
    
    def load_model_params(self, symbol: str, model_name: str, version: str = None) -> dict:
        """Load model parameters from mock storage."""
        try:
            if symbol not in self.model_params or model_name not in self.model_params[symbol]:
                return {}
            
            if version:
                # Load specific version
                if version in self.model_params[symbol][model_name]:
                    return self.model_params[symbol][model_name][version]["params"]
                return {}
            else:
                # Load active version
                for ver, ver_data in self.model_params[symbol][model_name].items():
                    if ver_data.get("status") == "active":
                        return ver_data["params"]
                return {}
        except Exception as e:
            print(f"Error loading model parameters for {symbol}/{model_name}: {e}")
            return {}
    
    def list_symbols(self) -> list:
        """List all available symbols in mock storage."""
        return list(self.stock_data.keys())
    
    def list_models(self, symbol: str) -> list:
        """List all available models for a symbol in mock storage."""
        if symbol not in self.model_params:
            return []
        return list(self.model_params[symbol].keys())
    
    def list_model_versions(self, symbol: str, model_name: str) -> list:
        """List all versions of a model with their details."""
        try:
            if symbol not in self.model_params or model_name not in self.model_params[symbol]:
                return []
            
            versions = []
            for ver, ver_data in self.model_params[symbol][model_name].items():
                version_info = {
                    "version": ver,
                    "timestamp": ver_data["timestamp"],
                    "status": ver_data["status"],
                    "description": ver_data["description"]
                }
                
                # Add metrics if available
                if ver_data.get("metrics"):
                    version_info.update(ver_data["metrics"])
                
                versions.append(version_info)
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return versions
        except Exception as e:
            print(f"Error listing model versions: {e}")
            return []
    
    def rollback_to_version(self, symbol: str, model_name: str, version: str) -> bool:
        """Rollback to a specific model version."""
        try:
            if symbol not in self.model_params or model_name not in self.model_params[symbol]:
                return False
            
            if version not in self.model_params[symbol][model_name]:
                return False
            
            # Mark current active version as superseded
            for ver, ver_data in self.model_params[symbol][model_name].items():
                if ver_data.get("status") == "active":
                    self.model_params[symbol][model_name][ver]["status"] = "superseded"
            
            # Create restored version
            restored_version = f"{version}_restored"
            self.model_params[symbol][model_name][restored_version] = {
                "params": self.model_params[symbol][model_name][version]["params"],
                "status": "active",
                "timestamp": datetime.now(),
                "description": f"Restored from version {version}",
                "metrics": self.model_params[symbol][model_name][version].get("metrics", {})
            }
            
            return True
        except Exception as e:
            print(f"Error rolling back model: {e}")
            return False
    
    def save_training_history(self, symbol: str, model_name: str, version: str, history: dict) -> bool:
        """Save model training history to mock storage."""
        try:
            # Create history entry
            if symbol not in self.model_history:
                self.model_history[symbol] = {}
            
            if model_name not in self.model_history[symbol]:
                self.model_history[symbol][model_name] = {}
            
            # Store history
            self.model_history[symbol][model_name][version] = history
            
            # Create point for the write API
            history_json = json.dumps(history, default=str)
            point = Point("model_training_history") \
                .tag("symbol", symbol) \
                .tag("model_name", model_name) \
                .tag("version", version) \
                .field("history", history_json)
            
            # Add summary metrics if available
            if "loss" in history and len(history["loss"]) > 0:
                point = point.field("final_loss", history["loss"][-1])
                point = point.field("epochs", len(history["loss"]))
            
            if "val_loss" in history and len(history["val_loss"]) > 0:
                point = point.field("final_val_loss", history["val_loss"][-1])
            
            # Write point to mock InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            
            return True
        except Exception as e:
            print(f"Error saving training history: {e}")
            return False
    
    def get_training_history(self, symbol: str, model_name: str, version: str) -> dict:
        """Get model training history from mock storage."""
        try:
            if symbol not in self.model_history or model_name not in self.model_history[symbol]:
                return {}
            
            if version in self.model_history[symbol][model_name]:
                return self.model_history[symbol][model_name][version]
            
            return {}
        except Exception as e:
            print(f"Error getting training history: {e}")
            return {}


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


def test_stock_data_storage(storage, symbol):
    """Test saving and loading stock data."""
    print(f"\n=== Testing stock data storage for {symbol} ===")
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(symbol)
    print(f"Sample data shape: {data.shape}")
    print(data.head())
    
    # Save data
    print("\nSaving data to mock InfluxDB...")
    success = storage.save_data(symbol, data)
    print(f"Save success: {success}")
    
    # Load data
    print("\nLoading data from mock InfluxDB...")
    loaded_data = storage.load_data(symbol)
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


def test_model_params_storage(storage, symbol):
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
    success = storage.save_model_params(
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
    success = storage.save_model_params(
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
    versions = storage.list_model_versions(symbol, model_name)
    for version in versions:
        print(f"Version: {version['version']}")
        print(f"  Status: {version['status']}")
        print(f"  Timestamp: {version['timestamp']}")
        print(f"  Description: {version['description']}")
        if 'accuracy' in version:
            print(f"  Accuracy: {version['accuracy']}")
        if 'loss' in version:
            print(f"  Loss: {version['loss']}")
    
    # Load active version (should be v2)
    print("\nLoading active model parameters...")
    loaded_params = storage.load_model_params(symbol, model_name)
    print(f"Loaded parameters: {loaded_params}")
    
    # Test rollback
    print("\nRolling back to version 1...")
    success = storage.rollback_to_version(symbol, model_name, "v1")
    print(f"Rollback success: {success}")
    
    # List versions after rollback
    print("\nListing model versions after rollback...")
    versions = storage.list_model_versions(symbol, model_name)
    for version in versions:
        print(f"Version: {version['version']}")
        print(f"  Status: {version['status']}")
    
    # Load active version after rollback (should be v1_restored)
    print("\nLoading active model parameters after rollback...")
    loaded_params = storage.load_model_params(symbol, model_name)
    print(f"Loaded parameters: {loaded_params}")
    
    return True


def test_training_history(storage, symbol):
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
    success = storage.save_training_history(symbol, model_name, version, history)
    print(f"Save success: {success}")
    
    # Load training history
    print("\nLoading training history...")
    loaded_history = storage.get_training_history(symbol, model_name, version)
    print(f"Loaded history keys: {list(loaded_history.keys())}")
    
    return True


def main():
    """Main function."""
    # Initialize mock InfluxDB storage
    print("Initializing mock InfluxDB storage...")
    storage = MockInfluxDBStorage()
    
    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        test_stock_data_storage(storage, symbol)
    
    # Test model parameters storage with versioning
    test_model_params_storage(storage, "AAPL")
    
    # Test training history storage
    test_training_history(storage, "AAPL")
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
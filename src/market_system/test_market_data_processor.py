"""
Test suite for market data processor module.
"""

import json
import pytest
from pathlib import Path
import pandas as pd
from datetime import datetime
import tempfile
import shutil

from market_data_processor import MarketDataProcessor, MarketDataError

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    return {
        "market_name": "test_market",
        "data": [
            {
                "timestamp": "2025-03-15T12:00:00Z",
                "price": 100.50,
                "volume": 1000,
                "additional_metrics": {
                    "volatility": 0.15,
                    "bid_ask_spread": 0.05
                }
            },
            {
                "timestamp": "2025-03-15T12:01:00Z",
                "price": 101.00,
                "volume": 1500,
                "additional_metrics": {
                    "volatility": 0.16,
                    "bid_ask_spread": 0.06
                }
            }
        ]
    }

@pytest.fixture
def market_data_files(temp_data_dir, sample_market_data):
    """Set up market data files for testing."""
    market_dir = temp_data_dir / "test_market"
    market_dir.mkdir()
    
    # Create two test files
    file1_path = market_dir / "data1.json"
    file2_path = market_dir / "data2.json"
    
    # First file uses the sample data
    with open(file1_path, 'w') as f:
        json.dump(sample_market_data, f)
    
    # Second file has different timestamps
    data2 = sample_market_data.copy()
    data2["data"][0]["timestamp"] = "2025-03-15T12:02:00Z"
    data2["data"][1]["timestamp"] = "2025-03-15T12:03:00Z"
    
    with open(file2_path, 'w') as f:
        json.dump(data2, f)
    
    return temp_data_dir

def test_initialization(temp_data_dir):
    """Test MarketDataProcessor initialization."""
    processor = MarketDataProcessor(temp_data_dir)
    assert processor.base_path == temp_data_dir

def test_initialization_invalid_path():
    """Test initialization with invalid path."""
    with pytest.raises(MarketDataError):
        MarketDataProcessor(Path("/nonexistent/path"))

def test_validate_json_file(market_data_files, sample_market_data):
    """Test JSON file validation."""
    processor = MarketDataProcessor(market_data_files)
    json_file = market_data_files / "test_market" / "data1.json"
    
    # Test valid file
    data = processor.validate_json_file(json_file)
    assert data == sample_market_data

def test_validate_json_file_invalid_schema(temp_data_dir):
    """Test validation with invalid schema."""
    processor = MarketDataProcessor(temp_data_dir)
    
    # Create invalid JSON file
    market_dir = temp_data_dir / "test_market"
    market_dir.mkdir()
    invalid_file = market_dir / "invalid.json"
    
    invalid_data = {
        "market_name": "test_market",
        "data": [
            {
                "timestamp": "invalid_timestamp",  # Invalid timestamp
                "price": -100,  # Invalid negative price
                "volume": "invalid_volume"  # Invalid volume type
            }
        ]
    }
    
    with open(invalid_file, 'w') as f:
        json.dump(invalid_data, f)
    
    with pytest.raises(MarketDataError):
        processor.validate_json_file(invalid_file)

def test_process_market_data(market_data_files):
    """Test market data processing."""
    processor = MarketDataProcessor(market_data_files)
    df = processor.process_market_data("test_market")
    
    # Check DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Combined data from both files
    assert list(df.columns) == [
        "timestamp", "price", "volume", "market_name", "source_file",
        "volatility", "bid_ask_spread"
    ]
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["price"])
    assert pd.api.types.is_numeric_dtype(df["volume"])
    
    # Verify chronological ordering
    assert df["timestamp"].is_monotonic_increasing

def test_process_market_data_nonexistent_market(market_data_files):
    """Test processing nonexistent market."""
    processor = MarketDataProcessor(market_data_files)
    with pytest.raises(MarketDataError):
        processor.process_market_data("nonexistent_market")

def test_get_market_summary(market_data_files):
    """Test market summary generation."""
    processor = MarketDataProcessor(market_data_files)
    summary = processor.get_market_summary("test_market")
    
    assert summary["market_name"] == "test_market"
    assert summary["data_points"] == 4
    assert all(k in summary["price_metrics"] for k in ["mean", "median", "min", "max"])
    assert all(k in summary["volume_metrics"] for k in ["total", "mean", "median"])
    
    # Verify date range
    assert datetime.fromisoformat(summary["date_range"]["start"].replace('Z', '+00:00'))
    assert datetime.fromisoformat(summary["date_range"]["end"].replace('Z', '+00:00'))

def test_duplicate_handling(market_data_files, sample_market_data):
    """Test handling of duplicate timestamps."""
    processor = MarketDataProcessor(market_data_files)
    
    # Create a file with duplicate timestamps
    dup_file = market_data_files / "test_market" / "duplicates.json"
    with open(dup_file, 'w') as f:
        json.dump(sample_market_data, f)  # Same timestamps as data1.json
    
    df = processor.process_market_data("test_market")
    
    # Check that duplicates were handled properly
    timestamp_counts = df["timestamp"].value_counts()
    assert all(count == 1 for count in timestamp_counts)
"""
Unit tests for the JSON normalizer.
"""

import pytest
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from market_system.json_normalizer import MarketDataNormalizer

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory structure with test data."""
    market_dir = tmp_path / "data" / "test_market" / "TEST.SYM"
    market_dir.mkdir(parents=True)
    
    test_data = {
        "open": 57.09,
        "high": 57.63,
        "low": 56.94,
        "close": 57.09,
        "volume": "2294978.45",
        "adjusted_close": "57.09000",
        "date": "2020-01-02",
        "currency": "USD",
        "metrics": {
            "returns": {
                "daily": -0.15,
                "intraday_range": 1.21
            },
            "indicators": {
                "vwap": "57.15",
                "turnover": 130920891.02
            }
        }
    }
    
    test_file = market_dir / "2020-01-02.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    return tmp_path

@pytest.fixture
def normalizer(test_data_dir):
    """Create normalizer instance with test directory."""
    return MarketDataNormalizer(test_data_dir)

def test_format_decimal(normalizer):
    """Test decimal formatting."""
    assert normalizer.format_decimal(57.09) == 57.0900
    assert normalizer.format_decimal("57.09") == 57.0900
    assert normalizer.format_decimal(57) == 57.0000
    assert normalizer.format_decimal(None) is None
    assert normalizer.format_decimal("invalid") is None

def test_format_integer(normalizer):
    """Test integer formatting."""
    assert normalizer.format_integer(2294978.45) == 2294978
    assert normalizer.format_integer("2294978") == 2294978
    assert normalizer.format_integer(None) is None
    assert normalizer.format_integer("invalid") is None

def test_format_date(normalizer):
    """Test date formatting."""
    assert normalizer.format_date("2020-01-02") == "2020-01-02"
    assert normalizer.format_date("20200102") == "2020-01-02"
    assert normalizer.format_date("02/01/2020") == "2020-01-02"
    assert normalizer.format_date(None) is None
    assert normalizer.format_date("invalid") is None

def test_normalize_data(normalizer):
    """Test data normalization."""
    input_data = {
        "open": 57.09,
        "high": 57.63,
        "low": 56.94,
        "close": 57.09,
        "volume": "2294978.45",
        "adjusted_close": "57.09000",
        "date": "2020-01-02",
        "currency": "USD",
        "metrics": {
            "returns": {
                "daily": -0.15,
                "intraday_range": 1.21
            },
            "indicators": {
                "vwap": "57.15",
                "turnover": 130920891.02
            }
        }
    }
    
    normalized = normalizer.normalize_data(input_data, "test_market", "TEST.SYM")
    
    assert normalized["open"] == 57.0900
    assert normalized["volume"] == 2294978
    assert normalized["date"] == "2020-01-02"
    assert normalized["market"] == "test_market"
    assert normalized["symbol"] == "TEST.SYM"
    assert normalized["metrics"]["returns"]["daily"] == -0.1500
    assert normalized["metrics"]["indicators"]["vwap"] == 57.1500

def test_process_file(normalizer, test_data_dir):
    """Test file processing."""
    test_file = test_data_dir / "data" / "test_market" / "TEST.SYM" / "2020-01-02.json"
    
    # Process the file
    normalizer.process_file(test_file)
    
    # Read and verify the processed file
    with open(test_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    assert processed_data["open"] == 57.0900
    assert processed_data["volume"] == 2294978
    assert processed_data["market"] == "test_market"
    assert processed_data["symbol"] == "TEST.SYM"
    assert processed_data["metrics"]["indicators"]["vwap"] == 57.1500

def test_process_all_files(normalizer, test_data_dir):
    """Test processing all files in directory structure."""
    normalizer.process_all_files()
    
    # Verify the test file was processed
    test_file = test_data_dir / "data" / "test_market" / "TEST.SYM" / "2020-01-02.json"
    assert test_file.exists()
    
    with open(test_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    assert processed_data["open"] == 57.0900
    assert processed_data["volume"] == 2294978

def test_invalid_base_dir():
    """Test initialization with invalid base directory."""
    with pytest.raises(ValueError):
        MarketDataNormalizer(Path("/nonexistent/path"))
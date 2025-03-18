"""
Test utilities for market system modules.
Provides common fixtures and helper functions for unit tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import os
from typing import Dict, List, Any, Generator, Tuple

# Test data generation

def generate_test_market_data(
    num_days: int = 30,
    symbols: List[str] = None,
    start_date: str = "2023-01-01",
    with_gaps: bool = False
) -> pd.DataFrame:
    """
    Generate test market data for unit tests.
    
    Args:
        num_days: Number of days of data
        symbols: List of symbols to generate data for
        start_date: Starting date for the data
        with_gaps: Whether to include gaps in the data
        
    Returns:
        pd.DataFrame: Generated market data
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Generate date range
    date_range = pd.date_range(start=start_date, periods=num_days)
    
    # Create empty data list
    data = []
    
    for symbol in symbols:
        base_price = np.random.uniform(50, 500)
        volatility = np.random.uniform(0.01, 0.05)
        
        for date in date_range:
            # Skip some dates if gaps enabled
            if with_gaps and np.random.random() < 0.1:
                continue
                
            # Generate random price data
            close = base_price * (1 + np.random.normal(0, volatility))
            open_price = close * (1 + np.random.normal(0, volatility * 0.5))
            high = max(close, open_price) * (1 + np.random.uniform(0, volatility))
            low = min(close, open_price) * (1 - np.random.uniform(0, volatility))
            volume = int(np.random.uniform(100000, 10000000))
            
            # Add data point
            data.append({
                "date": date,
                "symbol": symbol,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "adjusted_close": round(close, 2),
                "volume": volume,
                "market": "test"
            })
            
            # Update base price for next day
            base_price = close
    
    return pd.DataFrame(data)

def create_test_market_files(
    base_dir: Path,
    markets: List[str] = None,
    symbols_per_market: int = 3,
    days_per_symbol: int = 30
) -> Dict[str, List[str]]:
    """
    Create test market data files in a directory structure.
    
    Args:
        base_dir: Base directory to create files in
        markets: List of market names
        symbols_per_market: Number of symbols per market
        days_per_symbol: Number of days of data per symbol
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping markets to list of symbols
    """
    if markets is None:
        markets = ["nyse", "nasdaq"]
    
    market_symbols = {}
    
    for market in markets:
        # Create market directory
        market_dir = base_dir / market
        market_dir.mkdir(exist_ok=True)
        
        # Generate symbols
        symbols = [f"{market.upper()}{i:03d}" for i in range(symbols_per_market)]
        market_symbols[market] = symbols
        
        for symbol in symbols:
            # Create symbol directory
            symbol_dir = market_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Generate data
            start_date = "2023-01-01"
            date_range = pd.date_range(start=start_date, periods=days_per_symbol)
            
            base_price = np.random.uniform(50, 500)
            volatility = np.random.uniform(0.01, 0.05)
            
            for date in date_range:
                # Generate random price data
                close = base_price * (1 + np.random.normal(0, volatility))
                open_price = close * (1 + np.random.normal(0, volatility * 0.5))
                high = max(close, open_price) * (1 + np.random.uniform(0, volatility))
                low = min(close, open_price) * (1 - np.random.uniform(0, volatility))
                volume = int(np.random.uniform(100000, 10000000))
                
                # Create data object
                data = {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "adjusted_close": round(close, 2),
                    "volume": volume,
                    "market": market
                }
                
                # Write to file
                file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Update base price for next day
                base_price = close
    
    return market_symbols

# Pytest fixtures

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """
    Fixture providing sample market data as a DataFrame.
    
    Returns:
        pd.DataFrame: Sample market data
    """
    return generate_test_market_data()

@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary directory with test market data.
    
    Yields:
        Path: Path to test data directory
    """
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test data structure
        create_test_market_files(temp_dir)
        
        yield temp_dir
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

@pytest.fixture
def large_test_data_dir() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary directory with large test market data.
    
    Yields:
        Path: Path to test data directory
    """
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test data structure with more data
        create_test_market_files(
            temp_dir,
            markets=["nyse", "nasdaq", "asx", "hkex"],
            symbols_per_market=10,
            days_per_symbol=60
        )
        
        yield temp_dir
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, decimal: int = 4) -> bool:
    """
    Compare two DataFrames for approximate equality.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        decimal: Number of decimal places for floating point comparison
        
    Returns:
        bool: True if DataFrames are approximately equal
    """
    # Check shapes
    if df1.shape != df2.shape:
        return False
    
    # Check indices
    if not df1.index.equals(df2.index):
        return False
    
    # Check columns
    if set(df1.columns) != set(df2.columns):
        return False
    
    # Compare numeric columns with tolerance
    for col in df1.select_dtypes(include=['number']).columns:
        if not np.allclose(df1[col], df2[col], rtol=10**(-decimal), equal_nan=True):
            return False
    
    # Compare non-numeric columns exactly
    for col in df1.select_dtypes(exclude=['number']).columns:
        if not (df1[col] == df2[col]).all():
            return False
    
    return True 
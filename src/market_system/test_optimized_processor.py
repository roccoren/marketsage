"""
Test suite for optimized market data processor.
"""

import json
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import shutil
import time
from typing import Dict, List

from optimized_market_processor import OptimizedMarketProcessor, benchmark_processor

def generate_large_dataset(num_records: int) -> Dict:
    """Generate large test dataset with specified number of records."""
    data = {
        "market_name": "test_market",
        "data": []
    }
    
    base_time = pd.Timestamp.now()
    for i in range(num_records):
        data["data"].append({
            "timestamp": (base_time + pd.Timedelta(seconds=i)).isoformat(),
            "price": 100 + np.random.normal(0, 1),
            "volume": int(1000 + np.random.normal(0, 100)),
            "additional_metrics": {
                "volatility": 0.15 + np.random.normal(0, 0.01),
                "bid_ask_spread": 0.05 + np.random.normal(0, 0.01)
            }
        })
    
    return data

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def large_market_data(temp_data_dir):
    """Set up large market dataset for performance testing."""
    market_dir = temp_data_dir / "test_market"
    market_dir.mkdir()
    
    # Create multiple files with large datasets
    datasets = [
        generate_large_dataset(1000),  # 1K records
        generate_large_dataset(5000),  # 5K records
        generate_large_dataset(10000)  # 10K records
    ]
    
    for i, data in enumerate(datasets):
        file_path = market_dir / f"data_{i}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    return temp_data_dir

def test_memory_optimization(large_market_data):
    """Test memory usage optimization."""
    processor = OptimizedMarketProcessor(large_market_data)
    
    # Process data and get memory metrics
    df = processor.process_market_data("test_market", parallel=False)
    metrics = processor.get_performance_metrics()
    
    # Verify memory usage is tracked
    assert len(processor.performance.metrics['memory_usage']) > 0
    assert metrics['peak_memory_mb'] > 0
    assert metrics['avg_memory_mb'] > 0
    
    # Verify memory efficiency
    assert metrics['peak_memory_mb'] < 1000  # Should handle 16K records under 1GB

def test_parallel_processing_performance(large_market_data):
    """Test parallel vs sequential processing performance."""
    results = benchmark_processor(large_market_data, "test_market")
    
    # Verify both modes completed
    assert 'sequential' in results
    assert 'parallel' in results
    
    # On multi-core systems, parallel should be faster
    if processor_count := len(results['parallel']) > 1:
        assert (results['parallel']['records_per_second'] >
                results['sequential']['records_per_second'])

def test_chunked_processing(large_market_data):
    """Test chunked data processing."""
    processor = OptimizedMarketProcessor(large_market_data)
    
    # Process single file in chunks
    file_path = next((large_market_data / "test_market").glob("*.json"))
    df = processor.process_file(file_path)
    
    # Verify data integrity
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing
    assert pd.api.types.is_categorical_dtype(df["market_name"])

def test_performance_metrics_accuracy(large_market_data):
    """Test accuracy of performance metrics."""
    processor = OptimizedMarketProcessor(large_market_data)
    
    start_time = time.time()
    df = processor.process_market_data("test_market", parallel=True)
    total_time = time.time() - start_time
    
    metrics = processor.get_performance_metrics()
    
    # Verify timing accuracy
    assert abs((metrics['file_read_time_seconds'] + 
                metrics['processing_time_seconds'] - total_time)) < 1.0
    
    # Verify record count
    assert metrics['total_records'] == len(df)
    
    # Verify records per second calculation
    expected_rps = len(df) / total_time
    actual_rps = metrics['records_per_second']
    assert abs(expected_rps - actual_rps) / expected_rps < 0.1  # Within 10%

def test_memory_threshold_handling(temp_data_dir):
    """Test handling of memory threshold limits."""
    # Create very large dataset
    market_dir = temp_data_dir / "test_market"
    market_dir.mkdir()
    
    large_data = generate_large_dataset(50000)  # 50K records
    file_path = market_dir / "large_data.json"
    with open(file_path, 'w') as f:
        json.dump(large_data, f)
    
    processor = OptimizedMarketProcessor(temp_data_dir)
    df = processor.process_market_data("test_market", parallel=False)
    
    metrics = processor.get_performance_metrics()
    
    # Verify processing completed without memory errors
    assert len(df) == 50000
    assert metrics['peak_memory_mb'] > 0
    
def test_data_integrity_with_optimization(large_market_data):
    """Test that optimizations don't affect data integrity."""
    processor = OptimizedMarketProcessor(large_market_data)
    
    # Process same data sequentially and in parallel
    df_sequential = processor.process_market_data("test_market", parallel=False)
    df_parallel = processor.process_market_data("test_market", parallel=True)
    
    # Results should be identical
    pd.testing.assert_frame_equal(
        df_sequential.sort_index(axis=1),
        df_parallel.sort_index(axis=1)
    )
    
    # Verify no data was lost in chunked processing
    total_records = sum(
        len(json.load(open(f))["data"])
        for f in (large_market_data / "test_market").glob("*.json")
    )
    assert len(df_sequential) == total_records

def test_categorical_optimization(large_market_data):
    """Test memory optimization using categorical data types."""
    processor = OptimizedMarketProcessor(large_market_data)
    df = processor.process_market_data("test_market", parallel=False)
    
    # Verify categorical columns
    assert pd.api.types.is_categorical_dtype(df["market_name"])
    assert pd.api.types.is_categorical_dtype(df["source_file"])
    
    # Compare memory usage
    memory_categorical = df.memory_usage(deep=True).sum()
    
    # Convert to string and compare
    df_str = df.copy()
    df_str["market_name"] = df_str["market_name"].astype(str)
    df_str["source_file"] = df_str["source_file"].astype(str)
    memory_string = df_str.memory_usage(deep=True).sum()
    
    # Categorical should use less memory
    assert memory_categorical < memory_string
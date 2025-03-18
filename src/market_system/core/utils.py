"""
Utility functions for market data processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import os
import psutil
import time
import threading
import logging

logger = logging.getLogger(__name__)

# Memory and performance tracking utilities

def check_memory_usage(threshold: float = 0.75) -> bool:
    """
    Check if memory usage is below threshold.
    
    Args:
        threshold: Maximum memory usage threshold (0.0-1.0)
        
    Returns:
        bool: True if memory usage is below threshold
    """
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent / 100.0
    return memory_usage < threshold

def log_system_stats() -> Dict[str, float]:
    """
    Log and return system resource usage.
    
    Returns:
        Dict[str, float]: System statistics
    """
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    stats = {
        "memory_used_percent": memory_info.percent,
        "memory_available_gb": memory_info.available / (1024 ** 3),
        "cpu_usage_percent": cpu_usage
    }
    
    logger.info(f"System stats: Memory: {stats['memory_used_percent']:.1f}% CPU: {stats['cpu_usage_percent']:.1f}%")
    return stats

# Data processing utilities

def safe_series_value(value: Any, default: float = 0.0) -> float:
    """
    Safely extract a float value from a pandas Series or scalar value.
    Handles numpy types and series aggregations.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value
    """
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            value = value.iloc[0] if len(value) == 1 else value.mean()
        if isinstance(value, (np.generic, np.ndarray)):
            value = value.item() if hasattr(value, 'item') else float(value)
        return float(value) if pd.notnull(value) else default
    except (ValueError, TypeError, AttributeError):
        return default

def safe_float(value: Union[pd.Series, float, int, np.number], default: float = 0.0) -> float:
    """
    Safely convert various types to float.
    
    Args:
        value: Value to convert (can be Series, float, int, or numpy number)
        default: Default value to return if conversion fails
    
    Returns:
        float: Converted value or default
    """
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            # Extract a single value if possible
            value = value.iloc[0] if len(value) == 1 else value.mean()
        # Handle numpy types
        if isinstance(value, (np.generic, np.ndarray)):
            return float(value.item() if hasattr(value, 'item') else value)
        return float(value) if value is not None and pd.notnull(value) else default
    except (ValueError, TypeError, AttributeError):
        return default

def chunked_dataframe(df: pd.DataFrame, chunk_size: int = 100000) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks for processing.
    
    Args:
        df: DataFrame to split
        chunk_size: Number of rows per chunk
        
    Returns:
        List[pd.DataFrame]: List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size].copy())
    return chunks

# Thread-safe processing statistics tracker
class ProcessingStats:
    """Thread-safe processing statistics tracker."""
    
    def __init__(self):
        """Initialize statistics tracking."""
        self.start_time = time.time()
        self.files_processed = 0
        self.files_failed = 0
        self.current_batch = 0
        self.total_batches = 0
        self._lock = threading.Lock()

    def update(self, successful: bool) -> None:
        """
        Update processing statistics.
        
        Args:
            successful: Whether the file was processed successfully
        """
        with self._lock:
            if successful:
                self.files_processed += 1
            else:
                self.files_failed += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        elapsed_time = time.time() - self.start_time
        files_per_second = self.files_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "processed": self.files_processed,
            "failed": self.files_failed,
            "elapsed_seconds": elapsed_time,
            "files_per_second": files_per_second,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches
        } 
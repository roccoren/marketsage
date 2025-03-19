"""
Utility functions for Qlib and RDAgent integration.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Dict, Optional, Tuple

from qlib.data import D
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.contrib.report import analysis_position

def init_qlib(data_path: Optional[str] = None):
    """
    Initialize Qlib with proper configurations.
    
    Args:
        data_path: Path to Qlib data. If None, will use default.
    """
    # Default to local data path if not provided
    if data_path is None:
        data_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    
    # Initialize Qlib
    C.register_uri("qlib.workflow", "config")
    C.register_uri("qlib.workflow.task", "config.task")
    provider_uri = data_path
    
    C.set_conf(provider_uri=provider_uri)

def prepare_qlib_data(symbols: List[str], 
                     start_time: str, 
                     end_time: str, 
                     fields: Optional[List[str]] = None,
                     freq: str = 'day') -> pd.DataFrame:
    """
    Prepare data for Qlib model using Qlib's data API.
    
    Args:
        symbols: List of stock symbols
        start_time: Start time in format YYYY-MM-DD
        end_time: End time in format YYYY-MM-DD
        fields: Data fields to fetch. If None, will use default fields.
        freq: Data frequency
        
    Returns:
        DataFrame with prepared data
    """
    if fields is None:
        fields = [
            "($close-$open)/$open", 
            "($high-$low)/$open",
            "($volume-Mean($volume,5))/(Mean($volume,5))",
            "Mean($close, 5)/Mean($close, 10) - 1",
            "Mean($volume, 5)/Mean($volume, 10) - 1",
            "Std($close, 5)/Mean($close, 5)",
            "Std($volume, 5)/Mean($volume, 5)",
        ]
    
    # Initialize Qlib if not already initialized
    try:
        D.features
    except:
        init_qlib()
    
    data = D.features(
        symbols=symbols,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq
    )
    
    return data

def create_qlib_dataset(data: pd.DataFrame, 
                      target_field: str = 'Ref($close, -1)/$close - 1', 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create training, validation, and test datasets from Qlib data.
    
    Args:
        data: Input data from prepare_qlib_data
        target_field: Field to use as prediction target
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Add target field if needed
    if target_field not in data.columns:
        # We need to get the target data
        symbols = data.index.get_level_values('instrument').unique().tolist()
        start_time = data.index.get_level_values('datetime').min()
        end_time = data.index.get_level_values('datetime').max() + timedelta(days=5)  # Add buffer for future values
        
        target_data = D.features(
            symbols=symbols,
            fields=[target_field],
            start_time=start_time,
            end_time=end_time,
            freq="day"
        )
        
        # Merge with original data
        data = data.join(target_data, how='left')
    
    # Split by time
    times = sorted(data.index.get_level_values('datetime').unique())
    train_size = int(len(times) * train_ratio)
    val_size = int(len(times) * val_ratio)
    
    train_times = times[:train_size]
    val_times = times[train_size:train_size+val_size]
    test_times = times[train_size+val_size:]
    
    # Create datasets
    train_data = data.loc[data.index.get_level_values('datetime').isin(train_times)]
    val_data = data.loc[data.index.get_level_values('datetime').isin(val_times)]
    test_data = data.loc[data.index.get_level_values('datetime').isin(test_times)]
    
    return train_data, val_data, test_data 
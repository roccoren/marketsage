"""
Data loader module for Qlib integration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from qlib.data import D
from qlib.config import C
from qlib.utils import init_instance_by_config
from ..config import QLIB_CONFIG, DATA_HANDLER_CONFIG

class QlibDataLoader:
    """Data loader class for Qlib integration."""
    
    def __init__(self, config=None):
        """Initialize the data loader.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or DATA_HANDLER_CONFIG
        self._initialize_qlib()
        
    def _initialize_qlib(self):
        """Initialize Qlib with configuration."""
        import qlib
        qlib.init(**QLIB_CONFIG)
        
    def load_dataset(self, instruments=None, start_time=None, end_time=None):
        """Load dataset from Qlib.
        
        Args:
            instruments (str/list, optional): Stock symbols. Defaults to None.
            start_time (str, optional): Start time. Defaults to None.
            end_time (str, optional): End time. Defaults to None.
            
        Returns:
            tuple: (features, labels) tuple of pandas DataFrames
        """
        config = self.config.copy()
        if instruments:
            config["instruments"] = instruments
        if start_time:
            config["start_time"] = start_time
        if end_time:
            config["end_time"] = end_time
            
        handler = {
            "class": "Alpha360",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": config,
        }
        
        dataset = init_instance_by_config(handler)
        return dataset
    
    def get_features(self, instruments, fields, start_time=None, end_time=None):
        """Get specific features for instruments.
        
        Args:
            instruments (list): List of stock symbols
            fields (list): List of feature fields
            start_time (str, optional): Start time. Defaults to None.
            end_time (str, optional): End time. Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with requested features
        """
        start_time = start_time or self.config["start_time"]
        end_time = end_time or self.config["end_time"]
        
        data = D.features(
            instruments,
            fields,
            start_time=start_time,
            end_time=end_time,
            freq=QLIB_CONFIG["default_freq"]
        )
        return data
    
    def prepare_data_for_model(self, instruments=None):
        """Prepare data for model training/inference.
        
        Args:
            instruments (list, optional): List of stock symbols. Defaults to None.
            
        Returns:
            tuple: (train_dataset, valid_dataset) tuple of datasets
        """
        dataset = self.load_dataset(instruments)
        
        train_slice = slice(self.config["fit_start_time"], self.config["fit_end_time"])
        valid_slice = slice(self.config["fit_end_time"], self.config["end_time"])
        
        return dataset.prepare(train_slice, valid_slice) 
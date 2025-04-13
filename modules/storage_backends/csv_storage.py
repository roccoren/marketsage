#!/usr/bin/env python3
"""
CSV storage backend for the Qlib-based quantitative analysis project.
"""

import os
import json
from typing import Dict, List, Any, Optional

import pandas as pd

from modules.storage import StorageBackend


class CSVStorage(StorageBackend):
    """Storage backend using CSV files."""
    
    def __init__(self, data_dir: str = "data/csv"):
        """
        Initialize CSV storage.
        
        Args:
            data_dir (str): Directory for storing CSV files
        """
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, "stock_data")
        self.params_path = os.path.join(data_dir, "model_params")
        
        # Create directories if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.params_path, exist_ok=True)
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol to a CSV file.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        try:
            file_path = os.path.join(self.data_path, f"{symbol}.csv")
            data.to_csv(file_path)
            return True
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol from a CSV file.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_path = os.path.join(self.data_path, f"{symbol}.csv")
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            # Load data
            df = pd.read_csv(file_path, skiprows=[1,2], header=0, index_col=0, parse_dates=True)
            
            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_model_params(self, symbol: str, model_name: str, params: Dict[str, Any]) -> bool:
        """
        Save model parameters to a JSON file.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            params (Dict[str, Any]): Model parameters
            
        Returns:
            bool: Success status
        """
        try:
            # Create symbol directory if it doesn't exist
            symbol_dir = os.path.join(self.params_path, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save parameters
            file_path = os.path.join(symbol_dir, f"{model_name}.json")
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving model parameters for {symbol}/{model_name}: {e}")
            return False
    
    def load_model_params(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """
        Load model parameters from a JSON file.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model parameters
        """
        try:
            file_path = os.path.join(self.params_path, symbol, f"{model_name}.json")
            if not os.path.exists(file_path):
                return {}
            
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model parameters for {symbol}/{model_name}: {e}")
            return {}
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols based on CSV files.
        
        Returns:
            List[str]: List of symbols
        """
        try:
            files = os.listdir(self.data_path)
            return [f.split('.')[0] for f in files if f.endswith('.csv')]
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []
    
    def list_models(self, symbol: str) -> List[str]:
        """
        List all available models for a symbol based on JSON files.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[str]: List of model names
        """
        try:
            symbol_dir = os.path.join(self.params_path, symbol)
            if not os.path.exists(symbol_dir):
                return []
            
            files = os.listdir(symbol_dir)
            return [f.split('.')[0] for f in files if f.endswith('.json')]
        except Exception as e:
            print(f"Error listing models for {symbol}: {e}")
            return []
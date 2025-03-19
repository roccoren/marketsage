"""
Base strategy class for all trading strategies to inherit from
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from stockscan.utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class Strategy:
    """
    Base class for all trading strategies.
    
    This class provides common functionality for trading strategies,
    including data preparation, indicator calculation, and signal generation.
    All specific strategy implementations should inherit from this class.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.
        
        Parameters:
        -----------
        params : Dict[str, Any], optional
            Strategy parameters, by default None
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        logger.debug(f"Initialized {self.name} with params: {self.params}")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw price data
            
        Returns:
        --------
        pd.DataFrame
            Processed data
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Convert date index if it's not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime: {str(e)}")
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert column names to title case if they're lowercase or uppercase
        if not all(col in df.columns for col in required_columns):
            df.columns = [col.title() if col.lower() in [rc.lower() for rc in required_columns] else col 
                           for col in df.columns]
        
        # Check again after conversion
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing data
        df = df.dropna(subset=['Close'])
        
        # Fill remaining NaN values
        for col in ['Open', 'High', 'Low']:
            df[col] = df[col].fillna(df['Close'])
        df['Volume'] = df['Volume'].fillna(0)
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators based on strategy parameters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary of calculated indicators
        """
        # Use the technical indicators utility
        indicators = TechnicalIndicators.get_all_indicators(data, self.params)
        return indicators
    
    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze data and generate trading signals.
        
        This is an abstract method that must be implemented by subclasses.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of generated signals
            
        Raises:
        -------
        NotImplementedError
            If the method is not implemented by a subclass
        """
        raise NotImplementedError("Subclasses must implement analyze() method")
    
    def generate_signal(self, date: Union[str, datetime], signal_type: str, price: float,
                      strength: float = 1.0, stop_loss: Optional[float] = None, 
                      take_profit: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized signal dictionary.
        
        Parameters:
        -----------
        date : Union[str, datetime]
            Signal date
        signal_type : str
            Signal type ('buy', 'sell', 'exit')
        price : float
            Signal price
        strength : float, optional
            Signal strength (0.0-1.0), by default 1.0
        stop_loss : float, optional
            Stop loss price, by default None
        take_profit : float, optional
            Take profit price, by default None
        metadata : Dict[str, Any], optional
            Additional signal metadata, by default None
            
        Returns:
        --------
        Dict[str, Any]
            Standardized signal dictionary
        """
        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        signal = {
            'date': date,
            'type': signal_type.lower(),
            'price': float(price),
            'strength': float(strength),
            'strategy': self.name,
        }
        
        if stop_loss is not None:
            signal['stop_loss'] = float(stop_loss)
        
        if take_profit is not None:
            signal['take_profit'] = float(take_profit)
        
        if metadata is not None:
            signal['metadata'] = metadata
        
        return signal
    
    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization.
        
        This method can be overridden by subclasses to define their own parameter grid.
        
        Returns:
        --------
        Dict[str, List[Any]]
            Dictionary of parameter names and possible values
        """
        return {}
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return f"{self.name}(params={self.params})"

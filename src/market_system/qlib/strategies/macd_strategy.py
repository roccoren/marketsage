"""
MACD strategy implementation using Qlib.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import QlibStrategyBase
from qlib.data import D

class MACDStrategy(QlibStrategyBase):
    """MACD strategy implementation."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, **kwargs):
        """Initialize MACD strategy.
        
        Args:
            fast_period (int): Fast EMA period. Defaults to 12.
            slow_period (int): Slow EMA period. Defaults to 26.
            signal_period (int): Signal line period. Defaults to 9.
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_macd(self, close_prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicators.
        
        Args:
            close_prices (pd.Series): Series of close prices
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing MACD line, signal line, and histogram
        """
        # Calculate EMAs
        fast_ema = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram
        }
    
    def generate_signals(self, dataset) -> pd.Series:
        """Generate trading signals based on MACD crossovers.
        
        Args:
            dataset: Dataset containing price data
            
        Returns:
            pd.Series: Trading signals for each instrument
        """
        # Get close prices for all instruments
        instruments = dataset.instruments
        fields = ["$close"]
        close_prices = D.features(instruments, fields)
        
        signals = pd.Series(0.0, index=instruments)
        
        # Calculate signals for each instrument
        for instrument in instruments:
            instrument_prices = close_prices.loc[(slice(None), instrument), "$close"]
            macd_data = self.calculate_macd(instrument_prices)
            
            # Generate signal based on MACD histogram
            latest_histogram = macd_data["histogram"].iloc[-1]
            prev_histogram = macd_data["histogram"].iloc[-2]
            
            # Bullish signal: histogram crosses above zero
            if prev_histogram < 0 and latest_histogram > 0:
                signals[instrument] = 1.0
            # Bearish signal: histogram crosses below zero
            elif prev_histogram > 0 and latest_histogram < 0:
                signals[instrument] = -1.0
                
        return signals
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert MACD signals to position sizes.
        
        Args:
            signals (pd.Series): MACD signals
            
        Returns:
            pd.Series: Position sizes for each instrument
        """
        # Filter for only positive signals (long-only implementation)
        positive_signals = signals[signals > 0]
        
        if len(positive_signals) == 0:
            return pd.Series()
        
        # Use parent class position sizing with filtered signals
        return super().get_positions(positive_signals) 
"""
MACD Strategy module - Implementation of MACD trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from stockscan.strategies.base import Strategy
from stockscan.utils.technical_indicators import TechnicalIndicators

class SignalType(Enum):
    """Signal types for MACD strategy"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class MacdStrategy(Strategy):
    """
    Moving Average Convergence Divergence (MACD) strategy implementation.
    
    This strategy generates buy and sell signals based on MACD crossovers,
    with additional filters for RSI and volume.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize MACD strategy with parameters.
        
        Parameters:
        -----------
        params : Optional[Dict[str, Any]]
            Dictionary of parameters for the strategy
        """
        super().__init__(params or {})
        
        # Set strategy name
        self.name = "MACD"
        
        self.indicators = TechnicalIndicators()
        
        # MACD parameters
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
        
        # Additional filter parameters
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_overbought = self.params.get('rsi_overbought', 70)
        self.rsi_oversold = self.params.get('rsi_oversold', 30)
        self.volume_ma_period = self.params.get('volume_ma_period', 20)
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes price data to generate MACD signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of signal dictionaries with 'Date' and 'Type' keys
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Extract close price as Series
        close_price = df['close'].iloc[:, 0] if isinstance(df['close'], pd.DataFrame) else df['close']
        
        # Calculate EMAs on the Series, not DataFrame
        fast_ema = close_price.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close_price.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Set values in the DataFrame
        df['macd'] = macd_line
        
        # Calculate Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate RSI for filtering
        delta = close_price.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Get the exponential moving average
        roll_up = up.ewm(com=self.rsi_period-1, adjust=False).mean()
        roll_down = down.ewm(com=self.rsi_period-1, adjust=False).mean()
        
        # Calculate the RSI
        rs = roll_up / roll_down
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate volume moving average for volume confirmation
        volume = df['volume'].iloc[:, 0] if isinstance(df['volume'], pd.DataFrame) else df['volume']
        df['volume_ma'] = volume.rolling(window=self.volume_ma_period).mean()
        
        # Generate signals
        df['signal'] = SignalType.HOLD.value
        
        # Convert comparison columns to series if needed
        macd = df['macd'].iloc[:, 0] if isinstance(df['macd'], pd.DataFrame) else df['macd']
        macd_signal = df['macd_signal'].iloc[:, 0] if isinstance(df['macd_signal'], pd.DataFrame) else df['macd_signal']
        macd_hist = df['macd_hist'].iloc[:, 0] if isinstance(df['macd_hist'], pd.DataFrame) else df['macd_hist']
        rsi = df['rsi'].iloc[:, 0] if isinstance(df['rsi'], pd.DataFrame) else df['rsi']
        volume = df['volume'].iloc[:, 0] if isinstance(df['volume'], pd.DataFrame) else df['volume']
        volume_ma = df['volume_ma'].iloc[:, 0] if isinstance(df['volume_ma'], pd.DataFrame) else df['volume_ma']
        
        # Buy signal: MACD crosses above signal line, RSI not overbought
        buy_signal = (
            (macd > macd_signal) &          # MACD above signal line
            (macd_hist > 0) &               # MACD histogram is positive
            (rsi < self.rsi_overbought) &   # RSI not overbought
            (volume > volume_ma)            # Above average volume
        )
        
        # Sell signal: MACD crosses below signal line or RSI overbought
        sell_signal = (
            (macd < macd_signal) |          # MACD below signal line
            (macd_hist < 0) |               # MACD histogram is negative
            (rsi > self.rsi_overbought) |   # RSI overbought
            (rsi < self.rsi_oversold)       # RSI oversold
        )
        
        # Set signals
        df.loc[buy_signal, 'signal'] = SignalType.BUY.value
        df.loc[sell_signal, 'signal'] = SignalType.SELL.value
        
        # Convert DataFrame signals to list of dictionaries for backtester
        signals = []
        for date, row in df.iterrows():
            if row['signal'] != SignalType.HOLD.value:
                signals.append({
                    'Date': date,
                    'Type': row['signal']
                })
        
        return signals
    
    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization.
        
        Returns:
        --------
        Dict[str, List[Any]]
            Dictionary of parameter names and possible values
        """
        return {
            'fast_period': [8, 12, 14],
            'slow_period': [21, 26, 30],
            'signal_period': [7, 9, 11],
            'rsi_period': [7, 14, 21],
            'rsi_overbought': [65, 70, 75],
            'rsi_oversold': [25, 30, 35],
            'volume_ma_period': [10, 20, 30]
        }

# For backward compatibility
MACDStrategy = MacdStrategy 
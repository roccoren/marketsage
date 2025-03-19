"""
RSI Strategy

This strategy generates trading signals based on RSI (Relative Strength Index) values.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class RSIStrategy:
    """RSI Strategy."""
    
    def __init__(self, 
                 period: int = 14,
                 oversold_threshold: float = 30,
                 overbought_threshold: float = 70,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.05,
                 position_size: float = 0.1):
        """Initialize strategy parameters.
        
        Args:
            period: Period for RSI calculation
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            position_size: Position size as fraction of portfolio
        """
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Get gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['rsi'] < self.oversold_threshold, 'signal'] = 1
        data.loc[data['rsi'] > self.overbought_threshold, 'signal'] = -1
        
        # Generate entry/exit signals
        data['position'] = data['signal'].diff()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict with signals and metadata
        """
        # Calculate signals
        signals = self.calculate_signals(data)
        
        # Get the latest signal
        latest_signal = signals.iloc[-1]
        
        return {
            'action': 'buy' if latest_signal['position'] == 1 else 'sell' if latest_signal['position'] == -1 else 'hold',
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'metadata': {
                'rsi': latest_signal['rsi'],
                'signal': latest_signal['signal']
            }
        } 
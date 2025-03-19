"""
Moving Average Crossover Strategy

This strategy generates trading signals based on the crossover of two moving averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class MovingAverageCrossover:
    """Moving Average Crossover Strategy."""
    
    def __init__(self, 
                 fast_period: int = 10,
                 slow_period: int = 30,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.05,
                 position_size: float = 0.1):
        """Initialize strategy parameters.
        
        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            position_size: Position size as fraction of portfolio
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
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
        # Calculate moving averages
        data['fast_ma'] = data['Close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['Close'].rolling(window=self.slow_period).mean()
        
        # Calculate crossover signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
        
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
                'fast_ma': latest_signal['fast_ma'],
                'slow_ma': latest_signal['slow_ma'],
                'signal': latest_signal['signal']
            }
        } 
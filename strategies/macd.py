import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import Strategy

class MACDStrategy(Strategy):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name="MACD")
        self.params = params or {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'volume_ma_period': 5,
            'volume_threshold': 1.05  # 5% above average
        }

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD components."""
        close = data['Close']
        
        # Calculate EMAs
        exp1 = close.ewm(span=self.params['fast_period'], adjust=False).mean()
        exp2 = close.ewm(span=self.params['slow_period'], adjust=False).mean()
        
        # Calculate MACD components
        macd = exp1 - exp2  # DIF line
        signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()  # DEA line
        hist = 2 * (macd - signal)  # MACD histogram
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })

    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analyze price data for MACD signals.
        
        Returns list of dictionaries containing signal information.
        """
        if len(data) < self.params['slow_period'] + self.params['signal_period']:
            return []

        signals = []
        
        # Calculate MACD components
        macd_data = self.calculate_macd(data)
        
        # Calculate volume metrics
        volume = data['Volume']
        vol_ma = volume.rolling(window=self.params['volume_ma_period']).mean()
        volume_ratio = volume / vol_ma
        
        # Calculate price changes
        price_change_1d = data['Close'].pct_change()
        price_change_5d = data['Close'].pct_change(periods=5)
        
        # Look for signals in the last day's data
        last_idx = len(data) - 1
        if last_idx < 1:  # Need at least 2 days of data
            return []
            
        current_dif = macd_data['MACD'].iloc[-1]
        current_dea = macd_data['Signal'].iloc[-1]
        previous_dif = macd_data['MACD'].iloc[-2]
        previous_dea = macd_data['Signal'].iloc[-2]
        current_hist = macd_data['Histogram'].iloc[-1]
        
        # Check for buy signal conditions
        if (current_dif > 0 and  # DIF above zero (bullish momentum)
            previous_dif <= previous_dea and  # Previous DIF was below/at DEA
            current_dif > current_dea and  # Current DIF crossed above DEA
            volume_ratio.iloc[-1] > self.params['volume_threshold']):  # Volume confirmation
            
            # Calculate signal strength
            strength = self.calculate_signal_strength(
                macd_hist=current_hist,
                volume_ratio=volume_ratio.iloc[-1],
                price_change=price_change_1d.iloc[-1]
            )
            
            # Create signal record
            signal = {
                'Price': data['Close'].iloc[-1],
                'PriceChange1d': price_change_1d.iloc[-1],
                'PriceChange5d': price_change_5d.iloc[-1],
                'Volume': volume.iloc[-1],
                'AverageVolume': vol_ma.iloc[-1],
                'VolumeRatio': volume_ratio.iloc[-1],
                'MACD': current_dif,
                'Signal': current_dea,
                'Histogram': current_hist,
                'SignalStrength': strength,
                'Type': 'BUY'
            }
            
            signals.append(signal)
        
        return signals
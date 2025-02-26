from strategies.base import Strategy
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class OBVStrategy(Strategy):
    def __init__(self):
        super().__init__("On-Balance Volume Strategy")
        self.logger = logging.getLogger(__name__)
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        # Ensure data is properly aligned and has no gaps
        data = data.copy()  # Create a copy to avoid modifying original
        data = data.sort_index()
        
        # Handle duplicate indices before reindexing
        if data.index.duplicated().any():
            data = data[~data.index.duplicated(keep='last')]
            
        date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(date_range)
        data = data.ffill()  # Forward fill missing values
        data = data.dropna()  # Drop any remaining NaN values
        
        close_diff = data['Close'].diff()
        
        # Initialize OBV
        obv = pd.Series(0.0, index=data.index)
        
        # Handle first value
        if len(data) > 0:
            # Handle initial volume value carefully
            try:
                volume_val = data['Volume'].iloc[0]
                if isinstance(volume_val, pd.Series):
                    obv.iloc[0] = float(volume_val.iloc[0])
                else:
                    obv.iloc[0] = float(volume_val)
            except (ValueError, TypeError):
                # If conversion fails, try alternate methods
                volume_series = data['Volume'].iloc[0]
                if hasattr(volume_series, 'values'):
                    obv.iloc[0] = float(volume_series.values[0])
                else:
                    obv.iloc[0] = 0.0  # Default to zero if all else fails
            
            # Use vectorized operations for better performance and to avoid sequence errors
            pos_volume = data['Volume'].where(close_diff > 0, 0)
            neg_volume = data['Volume'].where(close_diff < 0, 0)
            zero_volume = data['Volume'].where(close_diff == 0, 0)
            
            # Calculate cumulative sum
            cumsum = (pos_volume - neg_volume).cumsum()
            
            # Update OBV
            obv = cumsum + obv.iloc[0]
                
        return obv

    def analyze(self, data: pd.DataFrame) -> Optional[List[Dict]]:
        """Analyze price data to generate trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of signal dictionaries or None if no signals
        """
        # Filter for last 7 trading days
        last_7_days = datetime.now() - timedelta(days=7)
        data = data[data.index >= last_7_days]
        self.logger.info(f"Analyzing last 7 days of data from {data.index[0]} to {data.index[-1]}")
        
        # Calculate OBV
        obv = self._calculate_obv(data)
        
        signals = []
        if len(obv) >= 2:  # Need at least 2 points to compare
            try:
                # First try direct float conversion
                current_obv = float(obv.iloc[-1])
                previous_obv = float(obv.iloc[-2])
                current_price = float(data['Close'].iloc[-1])
                previous_price = float(data['Close'].iloc[-2])
            except (ValueError, TypeError):
                try:
                    # Try numpy values if direct conversion fails
                    current_obv = float(obv.iloc[-1].values[0])
                    previous_obv = float(obv.iloc[-2].values[0])
                    current_price = float(data['Close'].iloc[-1].values[0])
                    previous_price = float(data['Close'].iloc[-2].values[0])
                except (ValueError, TypeError, AttributeError):
                    # Last resort - try numpy array conversion
                    current_obv = float(np.array(obv.iloc[-1]))
                    previous_obv = float(np.array(obv.iloc[-2]))
                    current_price = float(np.array(data['Close'].iloc[-1]))
                    previous_price = float(np.array(data['Close'].iloc[-2]))
            
            # Calculate volatility for position sizing
            price_std = data['Close'].pct_change().std()
            volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Calculate entry/exit levels based on recent volatility
            volatility_window = data['High'].rolling(window=5).max() - data['Low'].rolling(window=5).min()
            current_volatility = float(volatility_window.iloc[-1])
            
            # Check if OBV is rising along with price trends
            if current_obv > previous_obv and current_price > previous_price:
                entry_price = current_price
                stop_loss = entry_price - current_volatility * 0.5
                take_profit = entry_price + current_volatility
                
                signals.append({
                    "Strategy": self.name,
                    "Signal": "BUY",
                    "Timestamp": data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    "Strength": "Strong",
                    "Reason": f"Rising OBV ({current_obv:.0f} > {previous_obv:.0f}) confirms upward price trend ({current_price:.2f} > {previous_price:.2f})",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "volume_confirmation": volume_ratio > 1.0,
                    "trade_timing": {
                        "optimal_entry": "First hour of trading" if volume_ratio > 1.5 else "Scale in during day",
                        "valid_until": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    }
                })
            elif current_obv < previous_obv and current_price < previous_price:
                entry_price = current_price
                stop_loss = entry_price + current_volatility * 0.5
                take_profit = entry_price - current_volatility
                
                signals.append({
                    "Strategy": self.name,
                    "Signal": "SELL",
                    "Timestamp": data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    "Strength": "Strong",
                    "Reason": f"Falling OBV ({current_obv:.0f} < {previous_obv:.0f}) confirms downward price trend ({current_price:.2f} < {previous_price:.2f})",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "volume_confirmation": volume_ratio > 1.0,
                    "trade_timing": {
                        "optimal_entry": "Last hour of trading" if volume_ratio > 1.5 else "Scale in during day",
                        "valid_until": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    }
                })
                
        return signals if signals else None
    
    def generate_signal(self, data):
        self.logger.info("Evaluating OBV conditions for volume based trend confirmation")
        obv = data.get('obv')
        price = data.get('price')
        previous_obv = data.get('previous_obv')
        previous_price = data.get('previous_price')
        self.logger.debug(f"Current OBV: {obv}, Price: {price}, Previous OBV: {previous_obv}, Previous Price: {previous_price}")
        
        if None not in (obv, price, previous_obv, previous_price):
            # Check if OBV is rising along with price trends
            if obv > previous_obv and price > previous_price:
                self.logger.info("Rising OBV with increasing price: BUY signal")
                return "buy"
            elif obv < previous_obv and price < previous_price:
                self.logger.info("Declining OBV with decreasing price: SELL signal")
                return "sell"
        
        self.logger.info("No clear volume trend: hold position")
        return "hold"

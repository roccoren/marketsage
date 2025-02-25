from strategies.base import Strategy
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ATRMovingAverageStrategy(Strategy):
    def __init__(self):
        super().__init__("ATR/Moving Average Strategy")
        self.logger = logging.getLogger(__name__)

    def analyze(self, data: pd.DataFrame) -> Optional[List[Dict]]:
        """Analyze price data to generate trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of signal dictionaries or None if no signals
        """
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
        
        # Calculate ATR (14 periods)
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14, min_periods=1).mean()
        
        # Calculate 20-day Moving Average
        ma = data['Close'].rolling(window=20, min_periods=1).mean()
        
        signals = []
        if len(data) > 0:
            # Ensure values are scalar floats
            try:
                # First try direct conversion
                latest_price = float(data['Close'].iloc[-1])
                latest_ma = float(ma.iloc[-1])
                latest_atr = float(atr.iloc[-1])
            except (ValueError, TypeError):
                try:
                    # If that fails, try numpy values
                    latest_price = float(data['Close'].iloc[-1].values[0])
                    latest_ma = float(ma.iloc[-1].values[0])
                    latest_atr = float(atr.iloc[-1].values[0])
                except (ValueError, TypeError, AttributeError):
                    # If that fails too, try direct numpy conversion
                    latest_price = float(data['Close'].iloc[-1])
                    latest_ma = float(ma.iloc[-1])
                    latest_atr = float(atr.iloc[-1])
            
            # Convert to float before comparison to avoid Series ambiguity
            price_diff = float(latest_ma - latest_price)
            atr_value = float(latest_atr)
            
            if price_diff > atr_value:
                signals.append({
                    "Strategy": self.name,
                    "Signal": "BUY",
                    "Strength": "Strong",
                    "Reason": f"Price ({latest_price:.2f}) is below MA ({latest_ma:.2f}) by more than ATR ({latest_atr:.2f})"
                })
            elif float(latest_price) > float(latest_ma + latest_atr):
                signals.append({
                    "Strategy": self.name,
                    "Signal": "SELL",
                    "Strength": "Strong",
                    "Reason": f"Price ({latest_price:.2f}) is above MA ({latest_ma:.2f}) by more than ATR ({latest_atr:.2f})"
                })
                
        return signals if signals else None

    def generate_signal(self, data):
        self.logger.info("Evaluating ATR with Moving Average conditions")
        atr = data.get('atr')
        moving_average = data.get('moving_average')
        price = data.get('price')
        self.logger.debug(f"ATR: {atr}, Moving Average: {moving_average}, Price: {price}")
        if atr is not None and moving_average is not None and price is not None:
            # Adjust thresholds dynamically using ATR
            if price < moving_average - atr:
                self.logger.info("Price is sufficiently below Moving Average: BUY signal")
                return "buy"
            elif price > moving_average + atr:
                self.logger.info("Price is sufficiently above Moving Average: SELL signal")
                return "sell"
        self.logger.info("No clear signal detected: hold position")
        return "hold"

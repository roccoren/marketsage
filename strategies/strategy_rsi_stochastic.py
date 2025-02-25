from strategies.base import Strategy
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class RSIStochStrategy(Strategy):
    def __init__(self):
        super().__init__("RSI/Stochastic Strategy")
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
        
        # Ensure index is unique and sorted
        data = data.sort_index()
        data = data[~data.index.duplicated(keep='last')]
        
        # Create new date range without reindexing first
        min_date = data.index.min()
        max_date = data.index.max()
        
        if min_date is not pd.NaT and max_date is not pd.NaT:
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            data = data.reindex(date_range)
        data = data.ffill()  # Forward fill missing values
        data = data.dropna()  # Drop any remaining NaN values
        
        # Calculate RSI (14 periods)
        delta = data['Close'].diff()
        # Handle division by zero
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic (14,3,3)
        low_14 = data['Low'].rolling(window=14, min_periods=1).min()
        high_14 = data['High'].rolling(window=14, min_periods=1).max()
        
        # Calculate %K more safely
        stoch_denom = high_14 - low_14
        k = pd.Series(0.0, index=data.index)
        # Calculate stochastic values directly to avoid dimension issues
        close_min_low14 = data['Close'] - low_14
        stoch_values = pd.Series(0.0, index=data.index)
        
        # Use boolean masking to avoid dimension issues
        mask = (stoch_denom > 0)
        stoch_values[mask] = 100.0 * close_min_low14[mask] / stoch_denom[mask]
        k = stoch_values
        
        # Calculate %D with more periods initially to avoid gaps
        d = k.rolling(window=3, min_periods=3).mean()
        
        signals = []
        if not rsi.empty and not d.empty:
            # Convert to float to avoid Series comparison issues
            latest_rsi = float(rsi.iloc[-1])
            latest_stoch = float(d.iloc[-1])
            
            if latest_rsi > 70 and latest_stoch > 80:
                signals.append({
                    "Strategy": self.name,
                    "Signal": "SELL",
                    "Strength": "Strong",
                    "Reason": f"Overbought - RSI: {latest_rsi:.2f}, Stochastic: {latest_stoch:.2f}"
                })
            elif latest_rsi < 30 and latest_stoch < 20:
                signals.append({
                    "Strategy": self.name,
                    "Signal": "BUY",
                    "Strength": "Strong",
                    "Reason": f"Oversold - RSI: {latest_rsi:.2f}, Stochastic: {latest_stoch:.2f}"
                })
                
        return signals if signals else None
    
    def generate_signal(self, data):
        self.logger.info("Evaluating RSI and Stochastic conditions")
        rsi = data.get('rsi')
        stoch = data.get('stochastic')
        self.logger.debug(f"RSI value: {rsi}, Stochastic value: {stoch}")
        if rsi is not None and stoch is not None:
            if rsi > 70 and stoch > 80:
                self.logger.info("Overbought condition detected: SELL signal")
                return "sell"
            elif rsi < 30 and stoch < 20:
                self.logger.info("Oversold condition detected: BUY signal")
                return "buy"
        self.logger.info("No signal detected, hold position")
        return "hold"

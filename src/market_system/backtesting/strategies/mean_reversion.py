"""
Mean Reversion Strategy Implementation.

This module implements a mean reversion strategy based on RSI (Relative Strength Index).
"""

import pandas as pd
import numpy as np
from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState
from typing import List, Optional, Dict

class MeanReversionStrategy(BaseStrategy):
    """
    A mean reversion strategy based on RSI (Relative Strength Index).
    
    This strategy generates buy signals when the RSI falls below the oversold threshold,
    and sell signals when the RSI rises above the overbought threshold.
    """
    
    def __init__(self, name="MeanReversion", rsi_period=3, oversold_threshold=30, 
                 overbought_threshold=70, stop_loss_pct=0.05, take_profit_pct=0.1, 
                 position_size=1.0):
        """
        Initialize the mean reversion strategy.
        
        Args:
            name (str): Name of the strategy
            rsi_period (int): Period for RSI calculation
            oversold_threshold (int): RSI threshold for oversold condition (buy signal)
            overbought_threshold (int): RSI threshold for overbought condition (sell signal)
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            position_size (float): Position size as a fraction of portfolio value
        """
        super().__init__(name)
        self.rsi_period = rsi_period
        # More aggressive thresholds
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size = position_size
        self._required_indicators = ['rsi']
    
    def analyze(self, data: List[MarketData], market_state: MarketState) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: List of MarketData objects
            market_state: Current market state classification
            
        Returns:
            Signal object if a trading opportunity is identified, None otherwise
        """
        # Check if we have enough data
        if len(data) < self.rsi_period + 1:
            return None
        
        # Convert MarketData objects to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Calculate RSI if not already present
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Calculate price deviation from moving average
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        
        price_deviation = (df['close'].iloc[-1] / df['sma_20'].iloc[-1] - 1) * 100
        
        # Get current and previous RSI values
        current_rsi = df['rsi'].iloc[-1]
        previous_rsi = df['rsi'].iloc[-2]
        
        # Check for RSI trend (is it moving up or down)
        rsi_trend = current_rsi - previous_rsi
        
        # Create indicators snapshot
        indicators_snapshot = {
            'rsi': float(current_rsi),
            'rsi_trend': float(rsi_trend),
            'price_deviation': float(price_deviation),
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold
        }
        
        # Buy signal conditions:
        # 1. RSI crosses below oversold threshold OR
        # 2. RSI is below oversold threshold and starting to turn up OR
        # 3. Price is significantly below its moving average (oversold condition)
        buy_signal = (
            (current_rsi < self.oversold_threshold and previous_rsi >= self.oversold_threshold) or
            (current_rsi < self.oversold_threshold + 5 and rsi_trend > 1.0) or
            (price_deviation < -5.0 and current_rsi < 40)
        )
        
        if buy_signal:
            # Calculate confidence based on multiple factors
            rsi_factor = min(0.4, (self.oversold_threshold - current_rsi) / 15)
            deviation_factor = min(0.4, abs(price_deviation) / 10)
            trend_factor = min(0.2, abs(rsi_trend) / 5)
            
            confidence = min(0.9, 0.4 + rsi_factor + deviation_factor + trend_factor)
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="BUY",
                confidence=confidence,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        # Sell signal conditions:
        # 1. RSI crosses above overbought threshold OR
        # 2. RSI is above overbought threshold and starting to turn down OR
        # 3. Price is significantly above its moving average (overbought condition)
        sell_signal = (
            (current_rsi > self.overbought_threshold and previous_rsi <= self.overbought_threshold) or
            (current_rsi > self.overbought_threshold - 5 and rsi_trend < -1.0) or
            (price_deviation > 5.0 and current_rsi > 60)
        )
        
        if sell_signal:
            # Calculate confidence based on multiple factors
            rsi_factor = min(0.4, (current_rsi - self.overbought_threshold) / 15)
            deviation_factor = min(0.4, abs(price_deviation) / 10)
            trend_factor = min(0.2, abs(rsi_trend) / 5)
            
            confidence = min(0.9, 0.4 + rsi_factor + deviation_factor + trend_factor)
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="SELL",
                confidence=confidence,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        return None
    
    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate stop loss level for a given signal.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Stop loss price level
        """
        current_price = float(data[-1].close_price)
        
        if signal.action == "BUY":
            return current_price * (1 - self.stop_loss_pct)
        else:
            return current_price * (1 + self.stop_loss_pct)
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level for a given signal.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Take profit price level
        """
        current_price = float(data[-1].close_price)
        
        # More aggressive take profit targets
        if signal.action == "BUY":
            return current_price * (1 + self.take_profit_pct * 1.5)
        else:
            return current_price * (1 - self.take_profit_pct * 1.5)
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _convert_to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Convert a list of MarketData objects to a DataFrame.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            DataFrame with OHLCV data
        """
        df_data = []
        
        for item in data:
            row = {
                'timestamp': item.timestamp,
                'symbol': item.symbol,
                'open': float(item.open_price),
                'high': float(item.high_price),
                'low': float(item.low_price),
                'close': float(item.close_price),
                'volume': item.volume
            }
            
            # Add indicators if available
            if hasattr(item, 'indicators'):
                for key, value in item.indicators.items():
                    row[key] = value
                    
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            
        return df 
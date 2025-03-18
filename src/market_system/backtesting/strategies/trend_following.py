"""
Trend Following Strategy Implementation.

This module implements a simple trend following strategy based on moving averages.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Optional, Dict

from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    A trend following strategy based on moving average crossovers.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    def __init__(self, name="TrendFollowing", fast_period=2, slow_period=4):
        """
        Initialize the trend following strategy.
        
        Args:
            name (str): Name of the strategy
            fast_period (int): Period for the fast moving average
            slow_period (int): Period for the slow moving average
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._required_indicators = ['fast_ma', 'slow_ma']
        logger.info(f"Initialized {self.name} strategy with fast_period={fast_period}, slow_period={slow_period}")
    
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
        if len(data) < self.slow_period:
            logger.debug(f"Not enough data for {self.name} strategy. Need at least {self.slow_period} bars.")
            return None
        
        logger.info(f"Analyzing {len(data)} data points for {data[-1].symbol} at {data[-1].timestamp}")
        
        # Convert MarketData objects to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Get current and previous values
        current_fast_ma = df['fast_ma'].iloc[-1]
        current_slow_ma = df['slow_ma'].iloc[-1]
        prev_fast_ma = df['fast_ma'].iloc[-2]
        prev_slow_ma = df['slow_ma'].iloc[-2]
        
        # Calculate crossover
        current_cross = current_fast_ma - current_slow_ma
        prev_cross = prev_fast_ma - prev_slow_ma
        
        logger.info(f"Current MA values - Fast: {current_fast_ma:.2f}, Slow: {current_slow_ma:.2f}")
        logger.info(f"Previous MA values - Fast: {prev_fast_ma:.2f}, Slow: {prev_slow_ma:.2f}")
        
        # Create indicators snapshot
        indicators_snapshot = {
            'fast_ma': float(current_fast_ma),
            'slow_ma': float(current_slow_ma),
            'crossover': float(current_cross)
        }
        
        # Generate BUY signal on bullish crossover
        if current_cross > 0 and prev_cross <= 0:
            logger.info(f"BUY signal generated for {data[-1].symbol} at {data[-1].timestamp} (Bullish crossover)")
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="BUY",
                confidence=0.8,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        # Generate SELL signal on bearish crossover
        elif current_cross < 0 and prev_cross >= 0:
            logger.info(f"SELL signal generated for {data[-1].symbol} at {data[-1].timestamp} (Bearish crossover)")
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="SELL",
                confidence=0.8,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        logger.info(f"No signal generated for {data[-1].symbol} at {data[-1].timestamp}")
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
        # Get the most recent data point
        current_data = data[-1]
        
        # For buy signals, set stop loss below recent low
        if signal.action == "BUY":
            # Use the lowest low of the last N bars
            lookback = min(10, len(data))
            recent_lows = [float(d.low_price) for d in data[-lookback:]]
            stop_price = min(recent_lows) * 0.98  # 2% below the lowest recent low
            
        # For sell signals, set stop loss above recent high
        else:
            # Use the highest high of the last N bars
            lookback = min(10, len(data))
            recent_highs = [float(d.high_price) for d in data[-lookback:]]
            stop_price = max(recent_highs) * 1.02  # 2% above the highest recent high
            
        return float(stop_price)
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level for a given signal.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Take profit price level
        """
        # Get the most recent data point
        current_data = data[-1]
        current_price = float(current_data.close_price)
        
        # Calculate ATR if available
        atr = 0
        if hasattr(current_data, 'indicators') and 'atr_14' in current_data.indicators:
            atr = current_data.indicators['atr_14']
        else:
            # Simple volatility estimate if ATR not available
            lookback = min(14, len(data))
            highs = [float(d.high_price) for d in data[-lookback:]]
            lows = [float(d.low_price) for d in data[-lookback:]]
            ranges = [h - l for h, l in zip(highs, lows)]
            atr = sum(ranges) / len(ranges)
        
        # For buy signals, set take profit at 4x the risk (more aggressive)
        if signal.action == "BUY":
            stop_loss = self.get_stop_loss(data, signal)
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 4)
            
        # For sell signals, set take profit at 4x the risk (more aggressive)
        else:
            stop_loss = self.get_stop_loss(data, signal)
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 4)
            
        return float(take_profit)
    
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
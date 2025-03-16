"""
Moving Average Crossover Strategy Implementation.

This module implements a strategy that identifies opportunities when a faster moving average 
crosses above a slower moving average, combined with volume and price-to-VWAP filters.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Optional, Dict

from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    A strategy based on moving average crossovers with volume and VWAP filters.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, combined with increased volume and price above VWAP.
    """
    
    def __init__(self, name="MovingAverageCrossover", fast_period=10, slow_period=30, 
                 volume_threshold=1.5, vwap_filter=True):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            name (str): Name of the strategy
            fast_period (int): Period for the fast moving average
            slow_period (int): Period for the slow moving average
            volume_threshold (float): Minimum volume increase compared to average
            vwap_filter (bool): Whether to use VWAP as an additional filter
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_threshold = volume_threshold
        self.vwap_filter = vwap_filter
        self._required_indicators = ['fast_ma', 'slow_ma', 'vwap', 'volume_ratio']
        logger.info(f"Initialized {self.name} strategy with fast_period={fast_period}, "
                   f"slow_period={slow_period}, volume_threshold={volume_threshold}")
    
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
        if len(data) < self.slow_period + 2:
            logger.debug(f"Not enough data for {self.name} strategy. Need at least {self.slow_period + 2} bars.")
            return None
        
        logger.info(f"Analyzing {len(data)} data points for {data[-1].symbol} at {data[-1].timestamp}")
        
        # Convert MarketData objects to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Calculate VWAP
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        # Calculate volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Check for crossover
        current_fast_ma = df['fast_ma'].iloc[-1]
        current_slow_ma = df['slow_ma'].iloc[-1]
        prev_fast_ma = df['fast_ma'].iloc[-2]
        prev_slow_ma = df['slow_ma'].iloc[-2]
        
        # Check if fast MA crossed above slow MA
        crossover_up = prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma
        
        # Check if fast MA crossed below slow MA
        crossover_down = prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma
        
        # Check volume condition
        volume_condition = df['volume_ratio'].iloc[-1] >= self.volume_threshold
        
        # Check VWAP condition
        vwap_condition = True
        if self.vwap_filter:
            current_price = float(data[-1].close_price)
            current_vwap = df['vwap'].iloc[-1]
            vwap_condition = current_price > current_vwap
        
        # Create indicators snapshot
        indicators_snapshot = {
            'fast_ma': float(current_fast_ma),
            'slow_ma': float(current_slow_ma),
            'volume_ratio': float(df['volume_ratio'].iloc[-1]),
            'vwap': float(df['vwap'].iloc[-1]),
            'price': float(data[-1].close_price)
        }
        
        # Generate BUY signal if conditions are met
        if crossover_up and volume_condition and vwap_condition:
            logger.info(f"BUY signal generated for {data[-1].symbol} at {data[-1].timestamp}")
            
            # Calculate confidence based on strength of signals
            confirmations = [
                crossover_up,
                volume_condition,
                vwap_condition
            ]
            weights = [0.5, 0.3, 0.2]
            confidence = self._calculate_confidence(confirmations, weights)
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="BUY",
                confidence=confidence,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        # Generate SELL signal if conditions are met
        elif crossover_down:
            logger.info(f"SELL signal generated for {data[-1].symbol} at {data[-1].timestamp}")
            
            return Signal(
                symbol=data[-1].symbol,
                timestamp=data[-1].timestamp,
                action="SELL",
                confidence=0.7,
                strategy_name=self.name,
                market_state=market_state,
                indicators_snapshot=indicators_snapshot
            )
        
        return None
    
    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate stop loss level for a given signal.
        
        For buy signals, set stop loss at the recent low or a percentage below entry.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Stop loss price
        """
        if signal.action == "BUY":
            # Use the lowest low of the last N periods
            lookback = min(10, len(data))
            recent_lows = [float(d.low_price) for d in data[-lookback:]]
            recent_low = min(recent_lows)
            
            # Alternative: percentage-based stop loss
            entry_price = float(data[-1].close_price)
            percentage_stop = entry_price * 0.97  # 3% below entry
            
            # Use the higher of the two
            return max(recent_low, percentage_stop)
        else:
            # For sell signals, set stop loss above recent high
            lookback = min(10, len(data))
            recent_highs = [float(d.high_price) for d in data[-lookback:]]
            recent_high = max(recent_highs)
            
            # Alternative: percentage-based stop loss
            entry_price = float(data[-1].close_price)
            percentage_stop = entry_price * 1.03  # 3% above entry
            
            # Use the lower of the two
            return min(recent_high, percentage_stop)
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level for a given signal.
        
        For buy signals, set take profit at a fixed risk-reward ratio from entry.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Take profit price
        """
        entry_price = float(data[-1].close_price)
        stop_loss = self.get_stop_loss(data, signal)
        
        if signal.action == "BUY":
            # Risk-reward ratio of 1:2
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2)
            return take_profit
        else:
            # For sell signals
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2)
            return take_profit
    
    def _convert_to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Convert list of MarketData objects to pandas DataFrame.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'open': float(d.open_price),
                'high': float(d.high_price),
                'low': float(d.low_price),
                'close': float(d.close_price),
                'volume': d.volume
            }
            for d in data
        ])
        
        return df 
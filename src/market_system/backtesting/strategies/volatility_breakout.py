"""
Volatility Breakout Strategy Implementation.

This module implements a strategy that identifies breakout opportunities based on 
increased volatility and volume, targeting stocks that are breaking out of consolidation patterns.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Optional, Dict

from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    A strategy that identifies breakout opportunities based on volatility expansion.
    
    This strategy generates buy signals when price breaks out of a consolidation pattern
    with increased volatility and volume, indicating potential trend initiation.
    """
    
    def __init__(self, name="VolatilityBreakout", lookback_period=20, 
                 volatility_threshold=1.5, volume_threshold=1.5, 
                 consolidation_threshold=0.5):
        """
        Initialize the volatility breakout strategy.
        
        Args:
            name (str): Name of the strategy
            lookback_period (int): Period for calculating historical volatility
            volatility_threshold (float): Minimum volatility increase to trigger signal
            volume_threshold (float): Minimum volume increase compared to average
            consolidation_threshold (float): Maximum volatility during consolidation
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.consolidation_threshold = consolidation_threshold
        self._required_indicators = ['atr', 'atr_ratio', 'volume_ratio', 'price_change']
        logger.info(f"Initialized {self.name} strategy with lookback_period={lookback_period}, "
                   f"volatility_threshold={volatility_threshold}, volume_threshold={volume_threshold}")
    
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
        if len(data) < self.lookback_period + 10:
            logger.debug(f"Not enough data for {self.name} strategy. Need at least {self.lookback_period + 10} bars.")
            return None
        
        logger.info(f"Analyzing {len(data)} data points for {data[-1].symbol} at {data[-1].timestamp}")
        
        # Convert MarketData objects to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Calculate True Range
        df['tr'] = df.apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close'].shift(1)),
                abs(x['low'] - x['close'].shift(1))
            ),
            axis=1
        )
        
        # Calculate Average True Range (ATR)
        df['atr'] = df['tr'].rolling(window=self.lookback_period).mean()
        
        # Calculate ATR ratio (current ATR / previous ATR)
        df['atr_ratio'] = df['atr'] / df['atr'].shift(5)
        
        # Calculate volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=self.lookback_period).mean()
        
        # Calculate price change
        df['price_change'] = df['close'].pct_change()
        
        # Check for consolidation before breakout
        # Consolidation is defined as a period of low volatility
        consolidation_window = 5
        recent_atr_ratios = df['atr_ratio'].iloc[-consolidation_window-1:-1]
        was_consolidating = (recent_atr_ratios < self.consolidation_threshold).any()
        
        # Check for breakout conditions
        current_atr_ratio = df['atr_ratio'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        current_price_change = df['price_change'].iloc[-1]
        
        # Breakout conditions:
        # 1. Volatility expansion (ATR ratio > threshold)
        volatility_condition = current_atr_ratio > self.volatility_threshold
        
        # 2. Volume expansion (Volume ratio > threshold)
        volume_condition = current_volume_ratio > self.volume_threshold
        
        # 3. Price movement (positive for upside breakout)
        price_condition = current_price_change > 0
        
        # Create indicators snapshot
        indicators_snapshot = {
            'atr': float(df['atr'].iloc[-1]),
            'atr_ratio': float(current_atr_ratio),
            'volume_ratio': float(current_volume_ratio),
            'price_change': float(current_price_change),
            'was_consolidating': was_consolidating
        }
        
        # Generate BUY signal if conditions are met
        if volatility_condition and volume_condition and price_condition and was_consolidating:
            logger.info(f"BUY signal generated for {data[-1].symbol} at {data[-1].timestamp}")
            
            # Calculate confidence based on strength of signals
            confirmations = [
                volatility_condition,
                volume_condition,
                price_condition,
                was_consolidating
            ]
            weights = [0.4, 0.3, 0.2, 0.1]
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
        
        # Generate SELL signal for downside breakout
        elif volatility_condition and volume_condition and not price_condition and was_consolidating:
            logger.info(f"SELL signal generated for {data[-1].symbol} at {data[-1].timestamp}")
            
            # Calculate confidence based on strength of signals
            confirmations = [
                volatility_condition,
                volume_condition,
                not price_condition,
                was_consolidating
            ]
            weights = [0.4, 0.3, 0.2, 0.1]
            confidence = self._calculate_confidence(confirmations, weights)
            
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
        
        For breakout trades, set stop loss at the low of the breakout bar or based on ATR.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Stop loss price
        """
        # Convert data to DataFrame for easier calculations
        df = self._convert_to_dataframe(data)
        
        # Calculate ATR
        df['tr'] = df.apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close'].shift(1)),
                abs(x['low'] - x['close'].shift(1))
            ),
            axis=1
        )
        df['atr'] = df['tr'].rolling(window=self.lookback_period).mean()
        
        current_atr = df['atr'].iloc[-1]
        entry_price = float(data[-1].close_price)
        
        if signal.action == "BUY":
            # For buy signals, set stop loss at the low of the breakout bar or 1.5 ATR below entry
            # (reduced from 2 ATR to make it more conservative)
            breakout_low = float(data[-1].low_price)
            atr_stop = entry_price - (current_atr * 1.5)
            
            # Use the higher of the two
            stop_loss = max(breakout_low, atr_stop)
            
            # Ensure stop loss is not too far from entry (max 5%)
            min_stop = entry_price * 0.95
            stop_loss = max(stop_loss, min_stop)
            
            return stop_loss
        else:
            # For sell signals, set stop loss at the high of the breakout bar or 1.5 ATR above entry
            breakout_high = float(data[-1].high_price)
            atr_stop = entry_price + (current_atr * 1.5)
            
            # Use the lower of the two
            stop_loss = min(breakout_high, atr_stop)
            
            # Ensure stop loss is not too far from entry (max 5%)
            max_stop = entry_price * 1.05
            stop_loss = min(stop_loss, max_stop)
            
            return stop_loss
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level for a given signal.
        
        For breakout trades, set take profit based on ATR and risk-reward ratio.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Take profit price
        """
        # Convert data to DataFrame for easier calculations
        df = self._convert_to_dataframe(data)
        
        # Calculate ATR
        df['tr'] = df.apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close'].shift(1)),
                abs(x['low'] - x['close'].shift(1))
            ),
            axis=1
        )
        df['atr'] = df['tr'].rolling(window=self.lookback_period).mean()
        
        current_atr = df['atr'].iloc[-1]
        entry_price = float(data[-1].close_price)
        stop_loss = self.get_stop_loss(data, signal)
        
        if signal.action == "BUY":
            # Risk-reward ratio of 1:2 (reduced from 1:3 to be more realistic)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2)
            
            # Alternative: ATR-based take profit (reduced from 5 ATR to 3 ATR)
            atr_take_profit = entry_price + (current_atr * 3)
            
            # Use the lower of the two
            return min(take_profit, atr_take_profit)
        else:
            # For sell signals
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2)
            
            # Alternative: ATR-based take profit
            atr_take_profit = entry_price - (current_atr * 3)
            
            # Use the higher of the two
            return max(take_profit, atr_take_profit)
    
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
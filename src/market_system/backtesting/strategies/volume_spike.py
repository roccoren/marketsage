"""
Volume Spike Strategy Implementation.

This module implements a strategy that identifies trading opportunities based on 
significant increases in trading volume, which often precede price movements.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Optional, Dict

from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState

logger = logging.getLogger(__name__)


class VolumeSpikeStrategy(BaseStrategy):
    """
    A strategy that identifies trading opportunities based on volume spikes.
    
    This strategy generates signals when there is a significant increase in trading volume,
    which often precedes price movements. It combines volume analysis with price action
    to determine the direction of the potential move.
    """
    
    def __init__(self, name="VolumeSpike", lookback_period=20, 
                 volume_threshold=2.5, price_change_threshold=0.5,
                 min_consolidation_days=3):
        """
        Initialize the volume spike strategy.
        
        Args:
            name (str): Name of the strategy
            lookback_period (int): Period for calculating average volume
            volume_threshold (float): Minimum volume increase to trigger signal
            price_change_threshold (float): Minimum price change percentage
            min_consolidation_days (int): Minimum days of consolidation before spike
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.price_change_threshold = price_change_threshold
        self.min_consolidation_days = min_consolidation_days
        self._required_indicators = ['volume_ratio', 'price_change', 'avg_volume']
        logger.info(f"Initialized {self.name} strategy with lookback_period={lookback_period}, "
                   f"volume_threshold={volume_threshold}, price_change_threshold={price_change_threshold}")
    
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
        if len(data) < self.lookback_period + 5:
            logger.debug(f"Not enough data for {self.name} strategy. Need at least {self.lookback_period + 5} bars.")
            return None
        
        logger.info(f"Analyzing {len(data)} data points for {data[-1].symbol} at {data[-1].timestamp}")
        
        # Convert MarketData objects to DataFrame for easier analysis
        df = self._convert_to_dataframe(data)
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=self.lookback_period).mean()
        
        # Calculate volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['avg_volume']
        
        # Calculate price change
        df['price_change'] = df['close'].pct_change() * 100
        
        # Check for consolidation before volume spike
        # Consolidation is defined as a period of low price volatility
        consolidation_window = self.min_consolidation_days
        recent_price_changes = df['price_change'].iloc[-consolidation_window-1:-1].abs()
        was_consolidating = (recent_price_changes < self.price_change_threshold).all()
        
        # Check for volume spike conditions
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        current_price_change = df['price_change'].iloc[-1]
        
        # Volume spike condition
        volume_spike = current_volume_ratio > self.volume_threshold
        
        # Price movement condition
        significant_price_move = abs(current_price_change) > self.price_change_threshold
        
        # Determine direction of the signal based on price action
        price_direction = 1 if current_price_change > 0 else -1
        
        # Create indicators snapshot
        indicators_snapshot = {
            'volume_ratio': float(current_volume_ratio),
            'price_change': float(current_price_change),
            'avg_volume': float(df['avg_volume'].iloc[-1]),
            'was_consolidating': was_consolidating
        }
        
        # Generate signal if conditions are met
        if volume_spike and significant_price_move:
            # Calculate confidence based on strength of signals
            confirmations = [
                volume_spike,
                significant_price_move,
                was_consolidating
            ]
            weights = [0.5, 0.3, 0.2]
            confidence = self._calculate_confidence(confirmations, weights)
            
            if price_direction > 0:
                # BUY signal
                logger.info(f"BUY signal generated for {data[-1].symbol} at {data[-1].timestamp}")
                
                return Signal(
                    symbol=data[-1].symbol,
                    timestamp=data[-1].timestamp,
                    action="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    market_state=market_state,
                    indicators_snapshot=indicators_snapshot
                )
            else:
                # SELL signal
                logger.info(f"SELL signal generated for {data[-1].symbol} at {data[-1].timestamp}")
                
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
        
        For volume spike trades, set stop loss at the low/high of the spike bar.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Stop loss price
        """
        if signal.action == "BUY":
            # For buy signals, set stop loss at the low of the volume spike bar
            spike_low = float(data[-1].low_price)
            
            # Alternative: percentage-based stop loss
            entry_price = float(data[-1].close_price)
            percentage_stop = entry_price * 0.97  # 3% below entry
            
            # Use the higher of the two
            return max(spike_low, percentage_stop)
        else:
            # For sell signals, set stop loss at the high of the volume spike bar
            spike_high = float(data[-1].high_price)
            
            # Alternative: percentage-based stop loss
            entry_price = float(data[-1].close_price)
            percentage_stop = entry_price * 1.03  # 3% above entry
            
            # Use the lower of the two
            return min(spike_high, percentage_stop)
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level for a given signal.
        
        For volume spike trades, set take profit based on the average daily range.
        
        Args:
            data: List of MarketData objects
            signal: Signal object
            
        Returns:
            Take profit price
        """
        # Convert data to DataFrame for easier calculations
        df = self._convert_to_dataframe(data)
        
        # Calculate average daily range
        df['daily_range'] = df['high'] - df['low']
        avg_daily_range = df['daily_range'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        entry_price = float(data[-1].close_price)
        stop_loss = self.get_stop_loss(data, signal)
        
        if signal.action == "BUY":
            # Risk-reward ratio of 1:2
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2)
            
            # Alternative: range-based take profit
            range_take_profit = entry_price + (avg_daily_range * 1.5)
            
            # Use the lower of the two
            return min(take_profit, range_take_profit)
        else:
            # For sell signals
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2)
            
            # Alternative: range-based take profit
            range_take_profit = entry_price - (avg_daily_range * 1.5)
            
            # Use the higher of the two
            return max(take_profit, range_take_profit)
    
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
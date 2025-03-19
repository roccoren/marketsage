import logging
from decimal import Decimal
from typing import List, Dict, Optional, Any, Union

import pandas as pd
import numpy as np

from ...domain.models import MarketData, Signal, MarketState
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates signals based on the crossover of fast and slow moving averages.
    - Buy when fast MA crosses above slow MA (golden cross)
    - Sell when fast MA crosses below slow MA (death cross)
    
    The strategy also filters signals based on volume and trend confirmation.
    """
    
    def __init__(self, 
                fast_period: int = 10, 
                slow_period: int = 30,
                signal_threshold: float = 0.0,
                volume_threshold: float = 1.5,
                priority: int = 1):  # Higher priority (lower number)
        """
        Initialize the strategy with parameters.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            signal_threshold: Minimum difference between fast and slow MA to generate a signal
            volume_threshold: Minimum volume compared to average to confirm a signal
            priority: Priority level for this strategy (lower number = higher priority)
        """
        super().__init__(f"MAC_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_threshold = signal_threshold
        self.volume_threshold = volume_threshold
        self.priority = priority  # Add priority attribute
        
        logger.info(f"Initialized {self.name} with fast_period={fast_period}, slow_period={slow_period}, " 
                   f"signal_threshold={signal_threshold}, volume_threshold={volume_threshold}, priority={priority}")
    
    def analyze(self, 
               market_data: List[MarketData], 
               market_state: MarketState = MarketState.UNDEFINED) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            market_data: List of MarketData objects
            market_state: Current market state
            
        Returns:
            Signal object if a signal is generated, None otherwise
        """
        if not market_data or len(market_data) < self.slow_period + 5:
            return None
            
        try:
            # Extract data
            symbol = market_data[-1].symbol
            timestamp = market_data[-1].timestamp
            current_price = market_data[-1].close_price
            
            # Get closing prices
            closing_prices = [data.close_price for data in market_data]
            
            # Calculate moving averages
            fast_ma = self._calculate_moving_average(closing_prices, self.fast_period)
            slow_ma = self._calculate_moving_average(closing_prices, self.slow_period)
            
            # Get previous values for crossover detection
            prev_fast_ma = self._calculate_moving_average(closing_prices[:-1], self.fast_period)
            prev_slow_ma = self._calculate_moving_average(closing_prices[:-1], self.slow_period)
            
            # Calculate crossover
            current_diff = fast_ma - slow_ma
            previous_diff = prev_fast_ma - prev_slow_ma
            
            # Get volume for confirmation
            volumes = [data.volume for data in market_data]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            
            # Calculate volume confirmation
            volume_confirmation = current_volume > (avg_volume * self.volume_threshold)
            
            # Check for golden cross (buy signal)
            if previous_diff <= Decimal(str(self.signal_threshold)) and current_diff > Decimal(str(self.signal_threshold)):
                # Enhance with trend confirmation - check if price is above both MAs
                trend_confirmation = current_price > fast_ma and current_price > slow_ma
                
                # Additional volume confirmation for buy signals
                if volume_confirmation and trend_confirmation:
                    confidence = self._calculate_confidence(current_diff, volumes[-5:], avg_volume)
                    
                    # Calculate stop loss and take profit
                    # Use ATR or percentage-based stop loss, more conservative
                    stop_loss = Decimal("0.02")  # 2% stop loss
                    
                    # Use a reasonable take profit based on recent price action
                    take_profit = Decimal("0.05")  # 5% take profit
                    
                    # Set quantity based on available capital (to be calculated by portfolio manager)
                    quantity = None
                    
                    # Create signal with meta information
                    signal = Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        action="BUY",
                        confidence=confidence,
                        strategy_name=self.name,
                        market_state=market_state,
                        indicators_snapshot={
                            "fast_ma": float(fast_ma),
                            "slow_ma": float(slow_ma),
                            "price": float(current_price),
                            "volume": int(current_volume),
                            "avg_volume": float(avg_volume),
                            "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 0.0
                        },
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=quantity,
                        priority=self.priority  # Include priority in signal
                    )
                    
                    logger.info(f"Generated BUY signal for {symbol} at {timestamp} with confidence {confidence}")
                    return signal
            
            # Check for death cross (sell signal)
            elif previous_diff >= Decimal(str(self.signal_threshold)) and current_diff < Decimal(str(self.signal_threshold)):
                # Enhance with trend confirmation - check if price is below both MAs
                trend_confirmation = current_price < fast_ma and current_price < slow_ma
                
                if volume_confirmation and trend_confirmation:
                    confidence = self._calculate_confidence(abs(current_diff), volumes[-5:], avg_volume)
                    
                    # Calculate stop loss and take profit for short position
                    stop_loss = Decimal("0.02")  # 2% stop loss for short (price increases)
                    take_profit = Decimal("0.05")  # 5% take profit for short (price decreases)
                    
                    # Create signal
                    signal = Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        action="SELL",
                        confidence=confidence,
                        strategy_name=self.name,
                        market_state=market_state,
                        indicators_snapshot={
                            "fast_ma": float(fast_ma),
                            "slow_ma": float(slow_ma),
                            "price": float(current_price),
                            "volume": int(current_volume),
                            "avg_volume": float(avg_volume),
                            "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 0.0
                        },
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=None,
                        priority=self.priority  # Include priority in signal
                    )
                    
                    logger.info(f"Generated SELL signal for {symbol} at {timestamp} with confidence {confidence}")
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MovingAverageCrossoverStrategy.analyze: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_moving_average(self, data: List[Decimal], period: int) -> Decimal:
        """
        Calculate the simple moving average for a given period.
        
        Args:
            data: List of price data
            period: Period for the moving average
            
        Returns:
            Decimal moving average value
        """
        if len(data) < period:
            # Return the average of available data if not enough data
            return sum(data) / Decimal(len(data))
            
        # Calculate SMA for the period
        sma_subset = data[-period:]
        return sum(sma_subset) / Decimal(period)
    
    def _calculate_confidence(self, 
                            diff: Decimal, 
                            recent_volumes: List[int], 
                            avg_volume: float) -> Decimal:
        """
        Calculate confidence based on MA difference and volume confirmation.
        
        Args:
            diff: Difference between fast and slow MA
            recent_volumes: List of recent volume data
            avg_volume: Average volume
            
        Returns:
            Confidence value (0-1)
        """
        # Base confidence on the magnitude of the difference
        base_confidence = min(abs(diff) / Decimal("0.05"), Decimal("1.0"))
        
        # Volume confidence
        volume_ratio = sum(recent_volumes) / (len(recent_volumes) * avg_volume) if avg_volume > 0 else Decimal("0.5")
        volume_confidence = min(volume_ratio / Decimal(str(self.volume_threshold)), Decimal("1.0"))
        
        # Combine confidences (weighted average)
        combined_confidence = (base_confidence * Decimal("0.7")) + (volume_confidence * Decimal("0.3"))
        
        # Ensure the confidence is between 0 and 1
        return max(min(combined_confidence, Decimal("1.0")), Decimal("0.1")) 
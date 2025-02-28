import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import Strategy
from utils.technical_indicators import TechnicalIndicators
import logging

logger = logging.getLogger(__name__)

class MultiIndicatorStrategy(Strategy):
    def __init__(
        self,
        # Moving Average parameters
        sma_short_period: int = 8,     # Faster trend
        sma_long_period: int = 15,     # Medium trend
        ema_short_period: int = 5,     # Quick reaction
        ema_long_period: int = 10,     # Short trend
        # RSI parameters
        rsi_period: int = 14,
        rsi_oversold: int = 30,    # Standard levels
        rsi_overbought: int = 70,  # Standard levels
        # Bollinger Bands parameters
        bb_period: int = 20,
        bb_std: float = 2.0,
        # MACD parameters
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Strategy weighting parameters
        trend_weight: float = 0.35,     # Keep higher trend weight
        momentum_weight: float = 0.3,    # Keep momentum
        volatility_weight: float = 0.15, # Lower volatility
        macd_weight: float = 0.2,       # Standard MACD
        # Signal thresholds
        buy_threshold: float = 0.15,     # Lower thresholds
        sell_threshold: float = -0.15,   # Lower thresholds
        # Risk management parameters
        atr_period: int = 14,
        atr_multiplier: float = 2.5,    # Keep wider stops
        profit_target_multiplier: float = 3.0,  # Keep targets
        # Volatility parameters
        bb_threshold: float = 0.4,      # Lower BB threshold
        volume_factor: float = 1.5,     # Standard volume requirement
        trailing_stop_factor: float = 0.7  # Keep trailing factor
    ):
        super().__init__()
        
        # Store parameters
        self.params = {
            'sma_short_period': sma_short_period,
            'sma_long_period': sma_long_period,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'atr_period': atr_period
        }
        
        # Store weights and thresholds
        self.weights = {
            'trend': trend_weight,
            'momentum': momentum_weight,
            'volatility': volatility_weight,
            'macd': macd_weight
        }
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.atr_multiplier = atr_multiplier
        self.profit_target_multiplier = profit_target_multiplier
        self.bb_threshold = bb_threshold
        self.volume_factor = volume_factor
        self.trailing_stop_factor = trailing_stop_factor
        
        # State variables
        self.current_position = None
        self.trailing_stop = None
        self.entry_price = None
        self.last_atr = None
        self.highest_price = None
        self.prev_close = None
        self.prev_rsi = None
        self.trend_direction = None
        
        # Validate weights sum to 1
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Strategy weights must sum to 1.0")
        
        logger.debug("Strategy initialized with parameters: %s", self.params)
        logger.debug("Strategy weights: %s", self.weights)
        logger.debug(f"Signal thresholds - Buy: {buy_threshold}, Sell: {sell_threshold}")
        logger.debug(f"ATR parameters - Period: {atr_period}, Multiplier: {atr_multiplier}")

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on multiple technical indicators."""
        logger.debug(f"Analyzing data frame of shape {data.shape}")
        
        min_periods = max(
            self.params['sma_long_period'],
            self.params['ema_long_period'],
            self.params['macd_slow'] + self.params['macd_signal']
        )
        
        if len(data) < min_periods:
            logger.debug(f"Insufficient data for analysis (need {min_periods} periods)")
            return []
        
        try:
            # Calculate all indicators
            indicators = TechnicalIndicators.get_all_indicators(data, self.params)
            logger.debug("Successfully calculated technical indicators")
            
            # Get latest values
            latest = data.iloc[-1]
            close_price = latest['Close']
            volume = latest['Volume']
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = volume / avg_volume
            
            # Update ATR and state
            self.last_atr = indicators['atr'].iloc[-1]
            
            # Calculate signals
            signals = []
            
            # Check trailing stop and profit target if in position
            if self.current_position is not None:
                if self.current_position == 'LONG':
                    # Update highest price seen since entry
                    if self.highest_price is None or close_price > self.highest_price:
                        self.highest_price = close_price
                        # Adjust trailing stop based on new high
                        stop_distance = self.last_atr * self.atr_multiplier
                        # Adaptive stop loss - tighter as profit grows
                        if close_price > self.entry_price + stop_distance:
                            stop_distance *= self.trailing_stop_factor  # Tighter stop
                        self.trailing_stop = max(
                            self.highest_price - stop_distance,
                            self.trailing_stop if self.trailing_stop is not None else 0
                        )
                    
                    profit_target = self.entry_price + (self.last_atr * self.profit_target_multiplier)
                    
                    # Exit if price hits trailing stop or profit target
                    if close_price <= self.trailing_stop or close_price >= profit_target:
                        signals.append({
                            'Type': 'SELL',
                            'Price': close_price,
                            'Reason': 'Stop/Target hit'
                        })
                        logger.info(f"Exit signal - Price: {close_price:.2f}, "
                                  f"Stop: {self.trailing_stop:.2f}, "
                                  f"Target: {profit_target:.2f}")
                        self.current_position = None
                        self.trailing_stop = None
                        self.entry_price = None
                        self.highest_price = None
                        return signals
            
            # Calculate new entry signals
            trend_signal = self._get_trend_signal(indicators)
            momentum_signal = self._get_momentum_signal(indicators)
            volatility_signal = self._get_volatility_signal(indicators, close_price)
            macd_signal = self._get_macd_signal(indicators)
            
            # Update trend direction
            if trend_signal > 0.5:
                self.trend_direction = 'UP'
            elif trend_signal < -0.5:
                self.trend_direction = 'DOWN'
            
            # Combine signals using weights
            combined_signal = (
                trend_signal * self.weights['trend'] +
                momentum_signal * self.weights['momentum'] +
                volatility_signal * self.weights['volatility'] +
                macd_signal * self.weights['macd']
            )
            
            logger.debug(f"Signal components - Trend: {trend_signal:.2f}, "
                      f"Momentum: {momentum_signal:.2f}, "
                      f"Volatility: {volatility_signal:.2f}, "
                      f"MACD: {macd_signal:.2f}")
            logger.debug(f"Combined signal: {combined_signal:.2f}")
            
            # Generate trading signals with volatility breakout focus
            if self.current_position is None:
                if combined_signal >= self.buy_threshold:
                    # Check for BB breakout and volume confirmation
                    bb_pos = self._get_bb_position(indicators, close_price)
                    rsi = indicators['rsi'].iloc[-1]
                    
                    # Entry conditions:
                    # 1. BB breakout or approaching breakout
                    # 2. Sufficient volume
                    # 3. Not extremely overbought
                    # 4. Positive trend
                    if (bb_pos >= self.bb_threshold and
                        volume_ratio > self.volume_factor and
                        rsi < self.params['rsi_overbought'] and
                        self.trend_direction == 'UP'):
                        
                        signals.append({
                            'Type': 'BUY',
                            'Price': close_price
                        })
                        logger.info(f"Generated BUY signal at price {close_price:.2f} "
                                  f"(signal strength: {combined_signal:.2f})")
                        self.current_position = 'LONG'
                        self.entry_price = close_price
                        self.trailing_stop = close_price - (self.last_atr * self.atr_multiplier)
                        self.highest_price = close_price
            elif self.current_position == 'LONG':
                if combined_signal <= self.sell_threshold:
                    signals.append({
                        'Type': 'SELL',
                        'Price': close_price
                    })
                    logger.info(f"Generated SELL signal at price {close_price:.2f} "
                              f"(signal strength: {combined_signal:.2f})")
                    self.current_position = None
                    self.trailing_stop = None
                    self.entry_price = None
                    self.highest_price = None
            else:
                logger.debug(f"No signal generated (combined score: {combined_signal:.2f})")
                
            self.prev_close = close_price
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
            return []

    def _get_trend_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """Calculate trend signal component (-1 to 1)"""
        # Use both SMA and EMA crossovers
        sma_signal = indicators['sma_crossover'].iloc[-1]
        ema_signal = indicators['ema_crossover'].iloc[-1]
        
        logger.debug(f"Trend signals - SMA: {sma_signal:.2f}, EMA: {ema_signal:.2f}")
        
        # Weight EMA more for faster reaction
        return (sma_signal * 0.4 + ema_signal * 0.6)

    def _get_momentum_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """Calculate momentum signal component (-1 to 1)"""
        rsi = indicators['rsi'].iloc[-1]
        
        logger.debug(f"RSI value: {rsi:.2f}")
        
        if rsi <= self.params['rsi_oversold']:
            return 1.0  # Strong buy signal
        elif rsi >= self.params['rsi_overbought']:
            return -1.0  # Strong sell signal
        else:
            # Scale RSI between oversold and overbought levels
            scaled_rsi = 2 * (rsi - self.params['rsi_oversold']) / (
                self.params['rsi_overbought'] - self.params['rsi_oversold']
            ) - 1
            return -scaled_rsi  # Invert so higher RSI = more bearish

    def _get_volatility_signal(
        self,
        indicators: Dict[str, pd.Series],
        price: float
    ) -> float:
        """Calculate volatility signal component (-1 to 1)"""
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        
        # Calculate price position within bands
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return 0
            
        relative_position = (price - bb_middle) / (band_width / 2)
        logger.debug(f"Price position relative to Bollinger Bands: {relative_position:.2f}")
        
        # Non-linear scaling for breakouts
        if abs(relative_position) > 1:
            return np.sign(relative_position)
        return relative_position

    def _get_macd_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """Calculate MACD signal component (-1 to 1)"""
        # Use MACD crossover signal
        macd_signal = indicators['macd_crossover'].iloc[-1]
        logger.debug(f"MACD crossover signal: {macd_signal:.2f}")
        return macd_signal

    def _get_bb_position(
        self,
        indicators: Dict[str, pd.Series],
        price: float
    ) -> float:
        """Calculate position relative to Bollinger Bands"""
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        
        band_width = bb_upper - bb_lower
        if band_width == 0:
            return 0
            
        return (price - bb_middle) / (band_width / 2)
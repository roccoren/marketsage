from typing import Dict, Optional, Tuple
import numpy as np
import logging
from datetime import datetime
from ..domain.models import MarketState, MarketData

class MarketClassifier:
    def __init__(self, config: Dict = None):
        self._config = config or {
            'trend': {
                'sma_fast': 20,
                'sma_slow': 50,
                'min_slope': 0.01,
                'trend_strength_rsi': 0.6,
                'weights': {
                    'sma_relationship': 0.5,  # Weight for SMA crossover signal
                    'slope': 0.3,            # Weight for price slope
                    'rsi': 0.2               # Weight for RSI confirmation
                }
            },
            'range': {
                'bb_threshold': 0.8,
                'min_touches': 3,
                'max_slope': 0.005
            },
            'gap': {
                'min_gap_percent': 0.01,
                'volume_factor': 1.5
            },
            'consolidation': {
                'min_signal_strength': 0.4  # Minimum strength needed for conflicting signals
            }
        }

    def classify_market_state(self, data: list[MarketData]) -> Tuple[MarketState, float]:
        """
        Analyze market data and classify the current market state.
        
        Args:
            data: List of MarketData objects, ordered by time
            
        Returns:
            Tuple of (MarketState, confidence_score)
        """
        if not data or len(data) < self._config['trend']['sma_slow']:
            logging.warning(f"Insufficient data points. Required: {self._config['trend']['sma_slow']}, Got: {len(data) if data else 0}")
            return MarketState.UNDEFINED, 0.0

        # Check for gaps first
        gap_state, gap_confidence = self._detect_gap(data[-2:])
        if gap_state != MarketState.UNDEFINED:
            return gap_state, gap_confidence

        # Check for trending conditions with weighted scoring
        trend_state, trend_confidence, trend_signals = self._detect_trend(data)
        
        # Check for ranging conditions
        range_confidence = self._detect_range(data)
        if range_confidence > 0.6:  # Sufficient range evidence
            return MarketState.RANGING, range_confidence

        # Handle conflicting signals that might indicate consolidation
        if trend_state == MarketState.UNDEFINED and trend_signals['signal_strength'] >= self._config['consolidation']['min_signal_strength']:
            logging.debug(f"Detected consolidation with signal strength: {trend_signals['signal_strength']:.2f}")
            return MarketState.CONSOLIDATING, trend_signals['signal_strength']

        return trend_state, trend_confidence

    def _detect_trend(self, data: list[MarketData]) -> Tuple[MarketState, float, Dict]:
        """Detect trending market conditions using weighted scoring."""
        # Get relevant indicators
        closes = [d.close_price for d in data]
        sma_fast = f"sma_{self._config['trend']['sma_fast']}"
        sma_slow = f"sma_{self._config['trend']['sma_slow']}"
        
        current_data = data[-1]
        logging.debug(f"Processing trend detection for latest data point: {current_data.timestamp}")
        
        # Get indicator values
        sma_fast_value = current_data.indicators.get(sma_fast)
        sma_slow_value = current_data.indicators.get(sma_slow)
        rsi = current_data.indicators.get(f"rsi_14", 50)
        
        logging.debug(f"Available indicators: {list(current_data.indicators.keys())}")
        logging.debug(f"SMA Fast ({sma_fast}): {sma_fast_value}")
        logging.debug(f"SMA Slow ({sma_slow}): {sma_slow_value}")
        logging.debug(f"RSI: {rsi}")
        
        if not (sma_fast_value and sma_slow_value):
            logging.warning(f"Missing required SMAs. Fast: {sma_fast_value is not None}, Slow: {sma_slow_value is not None}")
            return MarketState.UNDEFINED, 0.0, {'signal_strength': 0.0}

        # Calculate trend slope
        price_changes = np.diff([d.close_price for d in data[-20:]])
        slope = np.mean(price_changes)
        logging.debug(f"Trend slope: {slope}, min_slope: {self._config['trend']['min_slope']}")

        # Calculate component scores
        sma_score = (sma_fast_value / sma_slow_value - 1) * 50  # Normalized SMA relationship
        slope_score = slope / self._config['trend']['min_slope']  # Normalized slope
        rsi_score = (rsi - 50) / 50  # Normalized RSI (-1 to 1)

        # Calculate weighted trend score
        weights = self._config['trend']['weights']
        trend_score = (
            abs(sma_score) * weights['sma_relationship'] +
            abs(slope_score) * weights['slope'] +
            abs(rsi_score) * weights['rsi']
        )

        logging.debug(f"Trend analysis - SMA Score: {sma_score:.2f}, Slope Score: {slope_score:.2f}, "
                     f"RSI Score: {rsi_score:.2f}, Total Score: {trend_score:.2f}")

        # Determine trend direction based on signals
        sma_bullish = sma_fast_value > sma_slow_value
        slope_bullish = slope > 0
        rsi_bullish = rsi > 50

        signals = {
            'sma_bullish': sma_bullish,
            'slope_bullish': slope_bullish,
            'rsi_bullish': rsi_bullish,
            'signal_strength': trend_score
        }

        # Check for strong trending signals
        if trend_score >= 0.7 and all([sma_bullish, slope_bullish, rsi_bullish]):
            logging.debug(f"Detected uptrend with score: {trend_score:.2f}")
            return MarketState.TRENDING_UP, trend_score, signals
        elif trend_score >= 0.7 and all([not sma_bullish, not slope_bullish, not rsi_bullish]):
            logging.debug(f"Detected downtrend with score: {trend_score:.2f}")
            return MarketState.TRENDING_DOWN, trend_score, signals

        logging.debug(f"No clear trend detected. Signal strength: {trend_score:.2f}")
        return MarketState.UNDEFINED, 0.0, signals

    def _detect_range(self, data: list[MarketData]) -> float:
        """Detect ranging market conditions and return confidence score."""
        # Get Bollinger Bands values
        current_data = data[-1]
        bb_upper = current_data.indicators.get('BBU_20_2.0')
        bb_lower = current_data.indicators.get('BBL_20_2.0')
        bb_middle = current_data.indicators.get('BBM_20_2.0')
        
        if not (bb_upper and bb_lower and bb_middle):
            logging.debug("Missing Bollinger Bands indicators")
            return 0.0

        # Count touches of bands
        upper_touches = 0
        lower_touches = 0
        bb_width = (bb_upper - bb_lower) / bb_middle

        for d in data[-20:]:
            if d.high_price >= bb_upper * 0.995:
                upper_touches += 1
            if d.low_price <= bb_lower * 1.005:
                lower_touches += 1

        # Calculate confidence based on multiple factors
        confidence = 0.0
        
        # Factor 1: Number of band touches
        min_touches = self._config['range']['min_touches']
        touch_score = min(1.0, (upper_touches + lower_touches) / (min_touches * 2))
        
        # Factor 2: Bollinger Band width (narrowing bands indicate potential range)
        width_score = max(0, 1 - bb_width)
        
        # Factor 3: Price slope (flat slope indicates ranging)
        price_changes = np.diff([d.close_price for d in data[-20:]])
        slope = abs(np.mean(price_changes))
        slope_score = max(0, 1 - slope / self._config['range']['max_slope'])

        # Combine factors with weights
        confidence = (
            touch_score * 0.4 +
            width_score * 0.3 +
            slope_score * 0.3
        )

        logging.debug(f"Range detection - Touches: {upper_touches + lower_touches}, "
                     f"Width: {bb_width}, Slope: {slope}")
        logging.debug(f"Range confidence: {confidence}")
        return confidence

    def _detect_gap(self, data: list[MarketData]) -> Tuple[MarketState, float]:
        """Detect price gaps between consecutive periods."""
        if len(data) < 2:
            return MarketState.UNDEFINED, 0.0

        prev_close = float(data[0].close_price)
        current_open = float(data[1].open_price)
        avg_volume = float(data[0].volume)
        
        gap_percent = (current_open - prev_close) / prev_close
        volume_ratio = float(data[1].volume) / avg_volume if avg_volume > 0 else 1.0

        min_gap = self._config['gap']['min_gap_percent']
        
        logging.debug(f"Gap analysis - Gap%: {gap_percent:.2%}, Volume ratio: {volume_ratio:.2f}")
        
        if gap_percent >= min_gap and volume_ratio >= self._config['gap']['volume_factor']:
            confidence = min(1.0, gap_percent / min_gap * 0.7 + volume_ratio * 0.3)
            return MarketState.GAP_UP, confidence
        
        elif gap_percent <= -min_gap and volume_ratio >= self._config['gap']['volume_factor']:
            confidence = min(1.0, abs(gap_percent) / min_gap * 0.7 + volume_ratio * 0.3)
            return MarketState.GAP_DOWN, confidence

        return MarketState.UNDEFINED, 0.0
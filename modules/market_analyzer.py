#!/usr/bin/env python3
"""
Market analysis module for the Qlib-based quantitative analysis project.
This module provides market condition detection and analysis functionality
to support dynamic parameter adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta

from modules.indicators import (
    MovingAverage, RelativeStrengthIndex, BollingerBands,
    AverageTrueRange, RiskMetrics, get_indicator_registry
)
from modules.logging_manager import get_logger

logger = get_logger(__name__)


class MarketCondition(Enum):
    """Enumeration of market conditions."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class MarketState:
    """Class representing market state analysis results."""
    
    def __init__(self):
        """Initialize market state."""
        self.condition = MarketCondition.NEUTRAL
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.volume_ratio = 1.0
        self.is_stable = True
        self.metrics = {}

    def to_dict(self) -> Dict:
        """Convert market state to dictionary."""
        return {
            'condition': self.condition.value,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'volume_ratio': self.volume_ratio,
            'is_stable': self.is_stable,
            'metrics': self.metrics
        }


class MarketAnalyzer:
    """
    Analyzes market conditions using technical indicators and statistical measures.
    """
    
    def __init__(self):
        """Initialize market analyzer with default thresholds."""
        # Volatility thresholds (annualized)
        self.high_volatility_threshold = 0.25  # 25%
        self.low_volatility_threshold = 0.10   # 10%
        
        # Trend thresholds
        self.strong_trend_threshold = 0.15     # 15%
        self.trend_threshold = 0.05            # 5%
        
        # Volume thresholds
        self.high_volume_ratio = 1.5           # 50% above average
        self.low_volume_ratio = 0.7            # 30% below average
        
        # Initialize technical indicators
        self.ma_50 = MovingAverage(window=50)
        self.ma_200 = MovingAverage(window=200)
        self.rsi = RelativeStrengthIndex(window=14)
        self.bb = BollingerBands(window=20)
        self.atr = AverageTrueRange(window=14)
    
    def analyze_market(self, data: pd.DataFrame, lookback_days: int = 252) -> MarketState:
        """
        Analyze market conditions using recent data.
        
        Args:
            data (pd.DataFrame): Historical price data
            lookback_days (int): Number of days to look back
            
        Returns:
            MarketState: Current market state analysis
        """
        # Use recent data only
        recent_data = data.tail(lookback_days).copy()
        
        state = MarketState()
        
        try:
            # Calculate core metrics
            state.volatility = self._calculate_volatility(recent_data)
            state.trend_strength = self._calculate_trend_strength(recent_data)
            state.volume_ratio = self._calculate_volume_ratio(recent_data)
            state.is_stable = state.volatility < self.high_volatility_threshold
            
            # Calculate additional metrics
            state.metrics = self._calculate_market_metrics(recent_data)
            
            # Determine market condition
            state.condition = self._determine_market_condition(state)
            
            logger.info(f"Market analysis completed: {state.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            # Return neutral state on error
            state.condition = MarketCondition.NEUTRAL
        
        return state
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        returns = data['Close'].pct_change()
        return returns.std() * np.sqrt(252)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using multiple indicators.
        Returns a normalized value between -1 and 1.
        """
        # Price trend
        price_change = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        
        # Moving average trends
        ma_50_values = self.ma_50.calculate(data)
        ma_200_values = self.ma_200.calculate(data)
        
        ma_trend = (ma_50_values.iloc[-1] / ma_200_values.iloc[-1]) - 1
        
        # RSI trend
        rsi_values = self.rsi.calculate(data)
        rsi_trend = (rsi_values.iloc[-1] - 50) / 50  # Normalize around neutral point
        
        # Combine trends with weights
        trend_strength = (
            0.4 * price_change +    # Recent price action
            0.4 * ma_trend +        # Long-term trend
            0.2 * rsi_trend        # Momentum
        )
        
        return np.clip(trend_strength, -1, 1)
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calculate volume ratio compared to historical average."""
        recent_volume = data['Volume'].tail(20).mean()  # Last month
        historical_volume = data['Volume'].mean()
        return recent_volume / historical_volume if historical_volume > 0 else 1.0
    
    def _calculate_market_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate additional market metrics."""
        returns = data['Close'].pct_change().dropna()
        
        metrics = {
            'var_95': RiskMetrics.calculate_var(returns, 0.95),
            'cvar_95': RiskMetrics.calculate_cvar(returns, 0.95),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(data['Close']),
            'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': RiskMetrics.calculate_sortino_ratio(returns)
        }
        
        # Add technical indicator signals
        bb_values = self.bb.calculate(data)
        atr_values = self.atr.calculate(data)
        
        last_close = data['Close'].iloc[-1]
        bb_middle = bb_values['BB_middle_20'].iloc[-1]
        bb_upper = bb_values['BB_upper_20'].iloc[-1]
        bb_lower = bb_values['BB_lower_20'].iloc[-1]
        
        metrics.update({
            'bb_position': (last_close - bb_middle) / (bb_upper - bb_lower),
            'atr_ratio': atr_values.iloc[-1] / last_close,
            'ma_50_200_ratio': (
                self.ma_50.calculate(data).iloc[-1] /
                self.ma_200.calculate(data).iloc[-1]
            )
        })
        
        return metrics
    
    def _determine_market_condition(self, state: MarketState) -> MarketCondition:
        """Determine overall market condition based on analyzed metrics."""
        # Check volatility conditions first
        if state.volatility >= self.high_volatility_threshold:
            return MarketCondition.HIGH_VOLATILITY
        elif state.volatility <= self.low_volatility_threshold:
            return MarketCondition.LOW_VOLATILITY
            
        # Check trend conditions
        if state.trend_strength >= self.strong_trend_threshold:
            return MarketCondition.STRONG_BULL
        elif state.trend_strength <= -self.strong_trend_threshold:
            return MarketCondition.STRONG_BEAR
        elif state.trend_strength >= self.trend_threshold:
            return MarketCondition.BULL
        elif state.trend_strength <= -self.trend_threshold:
            return MarketCondition.BEAR
            
        return MarketCondition.NEUTRAL


class MarketAwareParameterManager:
    """
    Manages parameter adjustments based on market conditions.
    """
    
    def __init__(self):
        """Initialize parameter manager."""
        self.market_analyzer = MarketAnalyzer()
        self.base_params = {
            'lookback_window': 252,
            'prediction_horizon': 10,
            'volatility_threshold': 0.2,
            'backtest_split': 0.7
        }
    
    def adjust_parameters(self, params: Dict, market_state: MarketState) -> Dict:
        """
        Adjust parameters based on market conditions.
        
        Args:
            params (Dict): Base parameters
            market_state (MarketState): Current market state
            
        Returns:
            Dict: Adjusted parameters
        """
        adjusted_params = params.copy()
        
        # Adjust based on market condition
        if market_state.condition in [MarketCondition.STRONG_BULL, MarketCondition.STRONG_BEAR]:
            # Use more historical data in strong trends
            adjusted_params['lookback_window'] = min(
                int(params['lookback_window'] * 1.5),
                756  # Max 3 years
            )
            # Extend prediction horizon in strong trends
            adjusted_params['prediction_horizon'] = min(
                int(params['prediction_horizon'] * 1.3),
                15
            )
        
        # Adjust for volatility
        if market_state.condition == MarketCondition.HIGH_VOLATILITY:
            # More conservative parameters in high volatility
            adjusted_params['volatility_threshold'] *= 1.2
            adjusted_params['backtest_split'] = 0.8  # More training data
            adjusted_params['prediction_horizon'] = max(
                int(params['prediction_horizon'] * 0.7),
                5
            )
        elif market_state.condition == MarketCondition.LOW_VOLATILITY:
            # More aggressive parameters in low volatility
            adjusted_params['volatility_threshold'] *= 0.8
            adjusted_params['backtest_split'] = 0.6  # More testing data
        
        # Additional adjustments based on metrics
        if not market_state.is_stable:
            # More conservative in unstable markets
            adjusted_params['lookback_window'] = min(
                int(params['lookback_window'] * 1.2),
                756
            )
        
        return adjusted_params
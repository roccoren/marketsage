"""
Market System - A modular stock market analysis and trading system.

This package provides a comprehensive framework for:
- Market data processing and analysis
- Dynamic market state classification
- Strategy implementation and signal generation
- Position and risk management
- Performance monitoring
"""

from market_system.trading_system import TradingSystem
from market_system.domain.models import (
    MarketState,
    MarketData,
    Position,
    Signal,
    RiskMetrics,
    StrategyPerformance
)
from market_system.strategy.base_strategy import BaseStrategy
from market_system.strategy.trend_following_strategy import TrendFollowingStrategy

__version__ = '0.1.0'
__author__ = 'Your Name'

__all__ = [
    'TradingSystem',
    'MarketState',
    'MarketData',
    'Position',
    'Signal',
    'RiskMetrics',
    'StrategyPerformance',
    'BaseStrategy',
    'TrendFollowingStrategy',
]
"""
Trading strategy implementations
"""

from src.market_system.backtesting.strategies.trend_following import TrendFollowingStrategy
from src.market_system.backtesting.strategies.mean_reversion import MeanReversionStrategy
from src.market_system.backtesting.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.market_system.backtesting.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.market_system.backtesting.strategies.volume_spike import VolumeSpikeStrategy

__all__ = [
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MovingAverageCrossoverStrategy',
    'VolatilityBreakoutStrategy',
    'VolumeSpikeStrategy'
] 
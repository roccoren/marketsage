"""
Market system package for financial market data processing and analysis.
Provides tools for data normalization, optimization, and strategy analysis.
"""

from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import core classes for easy access
from .core.base_processor import BaseProcessor, MarketDataError
from .core.normalizer import MarketDataNormalizer
from .market_data_processor import MarketDataProcessor
from .optimized_market_processor import OptimizedMarketProcessor
from .strategy_analyzer import StrategyAnalyzer

# Version info
__version__ = '1.0.0'

# Public API
__all__ = [
    'MarketDataProcessor',
    'OptimizedMarketProcessor',
    'MarketDataNormalizer',
    'StrategyAnalyzer',
    'MarketDataError',
    'create_processor',
    'create_normalizer',
    'create_analyzer'
]

def create_processor(base_path: str, optimized: bool = True) -> BaseProcessor:
    """
    Create a market data processor.
    
    Args:
        base_path: Path to market data directory
        optimized: Whether to use the optimized processor
        
    Returns:
        BaseProcessor: Market data processor instance
    """
    path = Path(base_path)
    if optimized:
        return OptimizedMarketProcessor(path)
    else:
        return MarketDataProcessor(path)

def create_normalizer(base_path: str) -> MarketDataNormalizer:
    """
    Create a market data normalizer.
    
    Args:
        base_path: Path to market data directory
        
    Returns:
        MarketDataNormalizer: Market data normalizer instance
    """
    return MarketDataNormalizer(Path(base_path))

def create_analyzer(base_path: str) -> StrategyAnalyzer:
    """
    Create a strategy analyzer.
    
    Args:
        base_path: Path to market data directory
        
    Returns:
        StrategyAnalyzer: Strategy analyzer instance
    """
    return StrategyAnalyzer(Path(base_path))
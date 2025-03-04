from .macd import MACDStrategy
from .crossover import PriceCrossoverStrategy
from .base import Strategy

__all__ = ['Strategy', 'MACDStrategy', 'PriceCrossoverStrategy']

# Initialize strategies package
# This file makes the directory a Python package
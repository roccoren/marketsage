"""
StockScan - Stock Market Analysis and Trading System

A comprehensive toolkit for stock market analysis, technical indicator calculation, strategy backtesting, and signal generation
"""

# Package information
__version__ = "0.1.0"
__author__ = "StockScan Development Team"

# Import subpackages
from . import core
from . import strategies
from . import utils
from . import scanner

# Specify public subpackages
__all__ = ["core", "strategies", "utils", "scanner"]

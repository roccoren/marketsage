"""
StockScan Strategies Package

Contains various trading strategy implementations, including technical analysis, statistical, and machine learning strategies
"""

# Import submodules
from . import technical
from . import statistical
from . import ml

# Specify public modules
__all__ = ["technical", "statistical", "ml"]

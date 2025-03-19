"""
StockScan - A LEAN-based trading and backtesting platform

This module provides tools for data collection, analysis, 
strategy development, and backtesting of trading algorithms.
"""

__version__ = "1.0.0"

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import key components
try:
    # Check if LEAN is available
    import clr
    from src.StockScan.Algorithm.BaseAlgorithm import BaseAlgorithm
    from src.StockScan.Algorithm.TrendFollowingAlgorithm import TrendFollowingAlgorithm
    from src.StockScan.Algorithm.MeanReversionAlgorithm import MeanReversionAlgorithm
    from src.StockScan.Engine.LeanEngine import LeanEngineAdapter
    from src.StockScan.Data.LeanDataHandler import LeanDataHandler, YahooFinanceProvider
    
    # Import interfaces
    from src.StockScan.Interfaces.IAlgorithm import IAlgorithm
    from src.StockScan.Interfaces.IData import IData, IDataProvider
    from src.StockScan.Interfaces.IEngine import IEngine, IBacktester, ILiveTrader
    
    LEAN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing LEAN components: {e}")
    logging.warning("Some features may not be available")
    LEAN_AVAILABLE = False

# Define package exports
__all__ = [
    'BaseAlgorithm',
    'TrendFollowingAlgorithm',
    'MeanReversionAlgorithm',
    'LeanEngineAdapter',
    'LeanDataHandler',
    'YahooFinanceProvider',
    'IAlgorithm',
    'IData',
    'IDataProvider',
    'IEngine',
    'IBacktester',
    'ILiveTrader',
    'LEAN_AVAILABLE'
]

def get_available_algorithms():
    """
    Get a list of available trading algorithms.
    
    Returns:
        List of algorithm names
    """
    if not LEAN_AVAILABLE:
        return []
    
    from src.StockScan.Engine.LeanEngine import LeanEngineAdapter
    engine = LeanEngineAdapter()
    return engine.get_available_algorithms()

def create_engine(data_folder=None, results_folder=None, config_path=None):
    """
    Create and initialize a backtesting engine.
    
    Args:
        data_folder: Path to data folder
        results_folder: Path to results folder
        config_path: Path to configuration file
        
    Returns:
        Initialized engine instance
    """
    if not LEAN_AVAILABLE:
        raise ImportError("LEAN framework is not available")
    
    from src.StockScan.Engine.LeanEngine import LeanEngineAdapter
    engine = LeanEngineAdapter(
        data_folder=data_folder,
        results_folder=results_folder,
        config_path=config_path
    )
    return engine

def create_data_handler(data_folder=None, cache_data=True):
    """
    Create and initialize a data handler.
    
    Args:
        data_folder: Path to data folder
        cache_data: Whether to cache data
        
    Returns:
        Initialized data handler
    """
    from src.StockScan.Data.LeanDataHandler import LeanDataHandler
    data_handler = LeanDataHandler(
        data_folder=data_folder,
        cache_data=cache_data
    )
    data_handler.Initialize()
    return data_handler 
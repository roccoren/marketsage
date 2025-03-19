"""
Market system package for financial market data processing and analysis.
Provides tools for data normalization, optimization, and strategy analysis using the LEAN framework.
"""

from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import core LEAN components
from .lean.lean_adapter import LeanAdapter, BacktestMode

# Version info
__version__ = '1.0.0'

# Public API
__all__ = [
    'LeanAdapter',
    'BacktestMode',
    'create_adapter'
]

def create_adapter(config_path: str = None, 
                 data_folder: str = None, 
                 results_folder: str = None) -> LeanAdapter:
    """
    Create a LEAN adapter.
    
    Args:
        config_path: Path to configuration file
        data_folder: Path to data folder
        results_folder: Path to results folder
        
    Returns:
        LeanAdapter: LEAN adapter instance
    """
    return LeanAdapter(
        config_path=config_path,
        mode=BacktestMode.LEAN,
        data_folder=data_folder,
        results_folder=results_folder
    )
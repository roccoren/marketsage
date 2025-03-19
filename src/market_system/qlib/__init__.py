"""
Qlib integration module for market system.
This module provides integration with Microsoft's Qlib framework for quantitative investing.
"""

from pathlib import Path

# Define base paths
QLIB_BASE_DIR = Path(__file__).parent
DATA_DIR = QLIB_BASE_DIR / "data"
MODELS_DIR = QLIB_BASE_DIR / "models"
STRATEGIES_DIR = QLIB_BASE_DIR / "strategies"
BACKTEST_DIR = QLIB_BASE_DIR / "backtesting"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, STRATEGIES_DIR, BACKTEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 
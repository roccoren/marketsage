"""
Prediction module for the stockscan package

This module provides functionality for predicting price targets and generating trading signals.
"""

from .price_targets import (
    calculate_price_targets,
    get_entry_exit_signals,
    predict_price_movement,
    analyze_stock_for_trading
)

__all__ = [
    'calculate_price_targets',
    'get_entry_exit_signals',
    'predict_price_movement',
    'analyze_stock_for_trading'
] 
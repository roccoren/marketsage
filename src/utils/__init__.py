"""
Utility functions for StockScan
"""

from .optimize import (
    ModelOptimizer,
    StrategyOptimizer,
    FeatureSelector,
    feature_importance_analysis,
    model_ensemble
)

# Make imports of common utility functions easier
try:
    from .common import init_qlib, init_mlflow, close_mlflow, log_config
except ImportError:
    # These functions might be in another module
    pass

from src.utils.mlflow_utils import (
    setup_mlflow, 
    create_nested_run, 
    log_extended_params, 
    log_dataset, 
    log_model_with_signature,
    log_prediction_results, 
    log_backtest_metrics, 
    find_best_runs, 
    compare_runs,
    create_comparison_chart
)

__all__ = [
    "init_qlib", "init_mlflow", "log_config", "close_mlflow",
    "setup_mlflow", "create_nested_run", "log_extended_params", 
    "log_dataset", "log_model_with_signature", "log_prediction_results", 
    "log_backtest_metrics", "find_best_runs", "compare_runs", 
    "create_comparison_chart"
] 
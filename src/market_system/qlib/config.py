"""
Qlib configuration module.
"""

import os
from pathlib import Path

# Base configuration
QLIB_CONFIG = {
    "provider_uri": str(Path(__file__).parent / "data"),
    "region": "us",
    "feature_root_dir": str(Path(__file__).parent / "features"),
    "cache_dir": str(Path(__file__).parent / "cache"),
    "logging_level": "INFO",
    "time_zone": "America/New_York",
    "kernels": 16,  # Number of processes for data processing
    "default_freq": "day",
}

# Model configuration
MODEL_CONFIG = {
    "model_type": "lightgbm",  # Default model type
    "model_kwargs": {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
    },
}

# Data handler configuration
DATA_HANDLER_CONFIG = {
    "start_time": "2010-01-01",
    "end_time": "2024-01-01",
    "fit_start_time": "2010-01-01",
    "fit_end_time": "2021-12-31",
    "instruments": "all",
    "learn_processors": [
        {"class": "DropnaProcessor"},
        {"class": "DropnaLabel"},
    ],
    "infer_processors": [
        {"class": "DropnaProcessor"},
    ],
}

# Feature configuration
FEATURE_CONFIG = {
    "price_volume_features": [
        "kbar",  # OHLCV features
        "technical",  # Technical indicators
    ],
    "custom_features": [],  # Add custom features here
}

# Backtest configuration
BACKTEST_CONFIG = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "signal",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
} 
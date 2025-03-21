import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Qlib config
QLIB_CONFIG = {
    "provider_uri": str(DATA_DIR),
    "expression_cache": None,
    "calendar_provider": "LocalCalendarProvider",
    "start_time": "2015-01-01",
    "end_time": "2023-12-31"
}

# MLflow config
MLFLOW_CONFIG = {
    "tracking_uri": str(BASE_DIR / "mlruns"),  # Local mlflow tracking
    "experiment_name": "qlib_experiment",
    "run_name": None  # Will be set dynamically
}

# Model config
MODEL_CONFIG = {
    "model": "Linear",  # One of Qlib's built-in models
    "model_kwargs": {},
    "dataset_kwargs": {
        "handler": "Alpha158",
        "segments": {
            "train": ("2015-01-01", "2020-12-31"),
            "valid": ("2021-01-01", "2021-06-30"),
            "test": ("2021-07-01", "2022-12-31")
        }
    }
}

# Strategy config
STRATEGY_CONFIG = {
    "strategy": "TopkDropoutStrategy",  # One of Qlib's built-in strategies
    "strategy_kwargs": {
        "topk": 10,
        "n_drop": 2
    }
}

# Backtest config
BACKTEST_CONFIG = {
    "start_time": "2021-07-01",
    "end_time": "2022-12-31",
    "account": 100000000,
    "benchmark": "000300",  # CSI 300 for China market
    "exchange_kwargs": {
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5
    }
}

# Data fetching config
FETCH_CONFIG = {
    "region": "cn",  # cn for China, us for US
    "target_dir": "qlib_data",
    "csv_path": None,  # If None, will use qlib's default data source
    "instruments": None,  # If None, will use the default instruments for the region
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "exists_skip": True,  # Skip if data already exists
    "max_workers": 16  # Number of parallel workers for downloading data
} 
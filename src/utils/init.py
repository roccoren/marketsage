import os
import mlflow
import qlib
from qlib.utils import init_instance_by_config
from datetime import datetime
from typing import Dict, Any, Optional

from src.config import QLIB_CONFIG, MLFLOW_CONFIG


def init_qlib(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize Qlib with the provided config or default config
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration for Qlib initialization
        
    Returns
    -------
    None
    """
    if config is None:
        config = QLIB_CONFIG
    
    print(f"Initializing Qlib with config: {config}")
    
    # Extract the parameters from the config
    provider_uri = config.get("provider_uri", "./data")
    expression_cache = config.get("expression_cache", None)
    calendar_provider = config.get("calendar_provider", "LocalCalendarProvider")
    start_time = config.get("start_time", "2015-01-01")
    end_time = config.get("end_time", "2023-12-31")
    region = config.get("region", "cn")
    
    # Initialize Qlib
    qlib.init(
        provider_uri=provider_uri,
        expression_cache=expression_cache,
        calendar_provider=calendar_provider,
        start_time=start_time,
        end_time=end_time
    )
    
    print("Qlib initialized successfully")


def init_mlflow(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Initialize MLflow tracking and create a new run
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration for MLflow initialization
        
    Returns
    -------
    str
        MLflow run id
    """
    if config is None:
        config = MLFLOW_CONFIG
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config["tracking_uri"])
    
    # Create or get experiment
    experiment_name = config["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Set run name if not provided
    run_name = config.get("run_name")
    if run_name is None:
        run_name = f"qlib_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run
    mlflow_run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name
    )
    
    print(f"MLflow initialized with experiment '{experiment_name}' and run '{run_name}'")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Run ID: {mlflow_run.info.run_id}")
    
    return mlflow_run.info.run_id


def log_config(
    qlib_config: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    strategy_config: Dict[str, Any] = None,
    backtest_config: Dict[str, Any] = None
) -> None:
    """
    Log configuration parameters to MLflow
    
    Parameters
    ----------
    qlib_config : Dict[str, Any]
        Qlib configuration
    model_config : Dict[str, Any]
        Model configuration
    strategy_config : Dict[str, Any]
        Strategy configuration
    backtest_config : Dict[str, Any]
        Backtest configuration
        
    Returns
    -------
    None
    """
    if qlib_config:
        mlflow.log_params({f"qlib_{k}": v for k, v in qlib_config.items() if isinstance(v, (str, int, float, bool))})
    
    if model_config:
        mlflow.log_params({f"model_{k}": v for k, v in model_config.items() if isinstance(v, (str, int, float, bool))})
    
    if strategy_config:
        mlflow.log_params({f"strategy_{k}": v for k, v in strategy_config.items() if isinstance(v, (str, int, float, bool))})
    
    if backtest_config:
        mlflow.log_params({f"backtest_{k}": v for k, v in backtest_config.items() if isinstance(v, (str, int, float, bool))})


def close_mlflow() -> None:
    """
    End the current MLflow run
    
    Returns
    -------
    None
    """
    mlflow.end_run() 
import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import joblib
import json
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

from src.config import MLFLOW_CONFIG


# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


def setup_mlflow(config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Advanced MLflow setup with experiment creation and run management
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration for MLflow initialization
        
    Returns
    -------
    Tuple[str, str]
        MLflow experiment_id, run_id
    """
    if config is None:
        config = MLFLOW_CONFIG
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config["tracking_uri"])
    
    # Create or get experiment
    experiment_name = config["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"created_timestamp": datetime.now().isoformat()}
        )
    else:
        experiment_id = experiment.experiment_id
    
    # Set run name if not provided
    run_name = config.get("run_name")
    if run_name is None:
        run_name = f"qlib_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start MLflow run with nested option
    parent_run_id = config.get("parent_run_id")
    mlflow_run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=parent_run_id is not None
    )
    
    print(f"MLflow initialized with experiment '{experiment_name}' and run '{run_name}'")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Run ID: {mlflow_run.info.run_id}")
    
    # Set important tags
    mlflow.set_tag("mlflow.user", os.environ.get("USER", "unknown"))
    mlflow.set_tag("run_timestamp", datetime.now().isoformat())
    
    return experiment_id, mlflow_run.info.run_id


def create_nested_run(parent_run_id: str, run_name: str) -> str:
    """
    Create a nested run within a parent run
    
    Parameters
    ----------
    parent_run_id : str
        ID of the parent run
    run_name : str
        Name for the nested run
    
    Returns
    -------
    str
        ID of the nested run
    """
    with mlflow.start_run(run_id=parent_run_id, nested=True) as nested_run:
        mlflow.set_tag("parent_run_id", parent_run_id)
        mlflow.set_tag("run_name", run_name)
        nested_run_id = nested_run.info.run_id
    
    return nested_run_id


def log_extended_params(
    params_dict: Dict[str, Any],
    prefix: Optional[str] = None,
) -> None:
    """
    Log parameters to MLflow with optional prefix and handling complex objects
    
    Parameters
    ----------
    params_dict : Dict[str, Any]
        Dictionary of parameters to log
    prefix : Optional[str]
        Prefix to add to parameter names
    """
    for key, value in params_dict.items():
        param_name = f"{prefix}_{key}" if prefix else key
        
        # Handle different parameter types
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(param_name, value)
        elif isinstance(value, (list, dict, tuple)):
            # Save complex parameters as JSON files
            param_file = f"params/{param_name}.json"
            os.makedirs(os.path.dirname(param_file), exist_ok=True)
            with open(param_file, 'w') as f:
                json.dump(value, f, indent=2, cls=NumpyEncoder)
            mlflow.log_artifact(param_file)
            # Also log a shortened representation as a parameter
            mlflow.log_param(f"{param_name}_type", f"{type(value).__name__}[artifact]")


def log_dataset(
    dataset: Any,
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log dataset information and metadata to MLflow
    
    Parameters
    ----------
    dataset : Any
        Dataset object or dataframe
    name : str
        Name of the dataset
    description : Optional[str]
        Description of the dataset
    metadata : Optional[Dict[str, Any]]
        Additional metadata to log
    """
    # Save metadata
    dataset_meta = metadata or {}
    
    if hasattr(dataset, 'shape'):
        dataset_meta['shape'] = str(dataset.shape)
    
    if isinstance(dataset, pd.DataFrame):
        # Log DataFrame information
        dataset_meta.update({
            'columns': list(dataset.columns),
            'dtypes': {col: str(dtype) for col, dtype in dataset.dtypes.items()},
            'index_type': str(type(dataset.index)),
            'memory_usage': int(dataset.memory_usage(deep=True).sum()),  # Convert to int to avoid NumPy types
            'sample_rows': int(len(dataset))  # Convert to int to avoid NumPy types
        })
        
        # Save dataset as CSV and parquet for different use cases
        csv_path = f"datasets/{name}.csv"
        parquet_path = f"datasets/{name}.parquet"
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Save sample of the dataset if it's large
        if len(dataset) > 1000:
            sample_df = dataset.sample(1000)
            sample_df.to_csv(csv_path)
            dataset.to_parquet(parquet_path)
            dataset_meta['is_sample'] = True
            dataset_meta['full_size'] = int(len(dataset))  # Convert to int to avoid NumPy types
            dataset_meta['sample_size'] = 1000
        else:
            dataset.to_csv(csv_path)
            dataset.to_parquet(parquet_path)
        
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(parquet_path)
    
    # Log metadata
    meta_path = f"datasets/{name}_metadata.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    
    if description:
        dataset_meta['description'] = description
    
    with open(meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2, cls=NumpyEncoder)
    
    mlflow.log_artifact(meta_path)
    
    # Add dataset tags
    mlflow.set_tag(f"dataset.{name}.logged", "true")
    if description:
        mlflow.set_tag(f"dataset.{name}.description", description[:250])  # Truncate if needed


def log_model_with_signature(
    model,
    artifact_path: str,
    sample_input: Optional[Any] = None,
    sample_output: Optional[Any] = None,
    conda_env: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a model with input and output signatures
    
    Parameters
    ----------
    model : Any
        Model object to log
    artifact_path : str
        Path where the model will be logged
    sample_input : Optional[Any]
        Sample input to the model
    sample_output : Optional[Any]
        Sample output from the model
    conda_env : Optional[Dict[str, Any]]
        Conda environment specification
    """
    # If sample input and output are provided, we can infer the signature
    if sample_input is not None and sample_output is not None:
        signature = infer_signature(sample_input, sample_output)
        mlflow.sklearn.log_model(
            model, 
            artifact_path, 
            signature=signature,
            conda_env=conda_env
        )
    else:
        mlflow.sklearn.log_model(
            model, 
            artifact_path,
            conda_env=conda_env
        )


def log_prediction_results(predictions: pd.DataFrame, name: str) -> None:
    """
    Log prediction results with visualizations
    
    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing predictions
    name : str
        Name for the prediction results
    """
    # Save predictions as CSV
    pred_path = f"predictions/{name}.csv"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    predictions.to_csv(pred_path)
    mlflow.log_artifact(pred_path)
    
    # Create visualization of prediction distribution
    if isinstance(predictions, pd.DataFrame):
        # Create a heatmap of predictions
        plt.figure(figsize=(12, 8))
        
        # If there are too many columns, sample them
        if predictions.shape[1] > 20:
            selected_cols = np.random.choice(predictions.columns, 20, replace=False)
            heat_data = predictions[selected_cols]
        else:
            heat_data = predictions
            
        sns.heatmap(heat_data.T, cmap='coolwarm', center=0, linewidths=0)
        plt.title(f"Prediction Heatmap - {name}")
        plt.xlabel("Time Points")
        plt.ylabel("Stocks")
        
        # Save the heatmap
        heatmap_path = f"predictions/{name}_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(heatmap_path)
        
        # Create prediction distribution plot
        plt.figure(figsize=(10, 6))
        # Flatten all predictions and plot histogram
        all_preds = predictions.values.flatten()
        sns.histplot(all_preds, kde=True)
        plt.title(f"Prediction Distribution - {name}")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        
        # Save the distribution plot
        dist_path = f"predictions/{name}_distribution.png"
        plt.savefig(dist_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(dist_path)


def log_backtest_metrics(
    metrics: Dict[str, float],
    portfolio_values: pd.Series,
    positions: Optional[pd.DataFrame] = None,
    trades: Optional[pd.DataFrame] = None,
    name: str = "backtest"
) -> None:
    """
    Log backtesting metrics and visualizations
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of backtesting metrics
    portfolio_values : pd.Series
        Series of portfolio values over time
    positions : Optional[pd.DataFrame]
        DataFrame of positions over time
    trades : Optional[pd.DataFrame]
        DataFrame of trades
    name : str
        Name for the backtest results
    """
    # Convert NumPy types to Python native types in metrics
    clean_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in metrics.items()}
    
    # Log metrics
    mlflow.log_metrics(clean_metrics)
    
    # Create directory for artifacts
    backtest_dir = f"backtests/{name}"
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Log portfolio value chart
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values.index, portfolio_values.values)
    plt.title(f"Portfolio Value - {name}")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    
    # Calculate drawdown
    drawdown = (portfolio_values / portfolio_values.cummax() - 1.0)
    
    # Add drawdown as a secondary axis
    ax2 = plt.gca().twinx()
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100, alpha=0.3, color='r')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(drawdown.min() * 100 * 1.5, 5)
    
    value_path = f"{backtest_dir}/portfolio_value.png"
    plt.savefig(value_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(value_path)
    
    # Save portfolio values as CSV
    portfolio_values.to_csv(f"{backtest_dir}/portfolio_values.csv")
    mlflow.log_artifact(f"{backtest_dir}/portfolio_values.csv")
    
    # Log positions heatmap if available
    if positions is not None:
        # If there are too many columns, sample them
        if positions.shape[1] > 30:
            selected_cols = np.random.choice(positions.columns, 30, replace=False)
            positions_sample = positions[selected_cols]
        else:
            positions_sample = positions
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(positions_sample.T, cmap='Greens', linewidths=0)
        plt.title(f"Positions Heatmap - {name}")
        plt.xlabel("Time")
        plt.ylabel("Stocks")
        
        positions_path = f"{backtest_dir}/positions_heatmap.png"
        plt.savefig(positions_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(positions_path)
        
        # Save positions as CSV
        positions.to_csv(f"{backtest_dir}/positions.csv")
        mlflow.log_artifact(f"{backtest_dir}/positions.csv")
    
    # Log trades information if available
    if trades is not None:
        trades.to_csv(f"{backtest_dir}/trades.csv")
        mlflow.log_artifact(f"{backtest_dir}/trades.csv")
        
        # Create trades visualization
        if len(trades) > 0:
            plt.figure(figsize=(12, 6))
            buy_trades = trades[trades['direction'] == 'buy']
            sell_trades = trades[trades['direction'] == 'sell']
            
            plt.scatter(buy_trades.index, buy_trades['price'], 
                      marker='^', color='g', s=50, label='Buy')
            plt.scatter(sell_trades.index, sell_trades['price'], 
                      marker='v', color='r', s=50, label='Sell')
            
            plt.title(f"Trades - {name}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            
            trades_path = f"{backtest_dir}/trades.png"
            plt.savefig(trades_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(trades_path)


def find_best_runs(
    experiment_name: str,
    metric_name: str,
    max_results: int = 5,
    higher_is_better: bool = True
) -> List[Dict[str, Any]]:
    """
    Find the best runs in an experiment based on a metric
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    metric_name : str
        Name of the metric to optimize
    max_results : int
        Maximum number of results to return
    higher_is_better : bool
        Whether higher values of the metric are better
    
    Returns
    -------
    List[Dict[str, Any]]
        List of runs with their metrics and parameters
    """
    client = MlflowClient()
    
    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    experiment_id = experiment.experiment_id
    
    # Set order based on higher_is_better
    order = "DESC" if higher_is_better else "ASC"
    
    # Search for runs - don't use filter, just get all runs and filter afterwards
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=100,  # Get more runs and filter later
        run_view_type=ViewType.ACTIVE_ONLY
    )
    
    # Filter runs that have the specified metric
    valid_runs = [
        run for run in runs 
        if metric_name in run.data.metrics
    ]
    
    # Sort runs based on the metric
    if higher_is_better:
        valid_runs.sort(key=lambda x: x.data.metrics.get(metric_name, 0), reverse=True)
    else:
        valid_runs.sort(key=lambda x: x.data.metrics.get(metric_name, 0))
    
    # Limit to max_results
    valid_runs = valid_runs[:max_results]
    
    # Extract run information
    result = []
    for run in valid_runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
            metric_name: run.data.metrics.get(metric_name),
            "parameters": {k: v for k, v in run.data.params.items()},
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
            "artifact_uri": run.info.artifact_uri
        }
        result.append(run_info)
    
    return result


def compare_runs(
    run_ids: List[str],
    metric_names: List[str]
) -> pd.DataFrame:
    """
    Compare multiple runs based on specified metrics
    
    Parameters
    ----------
    run_ids : List[str]
        List of run IDs to compare
    metric_names : List[str]
        List of metric names to compare
    
    Returns
    -------
    pd.DataFrame
        DataFrame with runs as rows and metrics as columns
    """
    client = MlflowClient()
    
    # Initialize result list
    results = []
    
    # Fetch data for each run
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
                "model_type": run.data.params.get("model_type", "Unknown"),
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add metrics
            for metric_name in metric_names:
                metric_value = run.data.metrics.get(metric_name, None)
                # Convert NumPy types to native Python types
                if isinstance(metric_value, (np.floating, np.integer)):
                    metric_value = float(metric_value)
                run_data[metric_name] = metric_value
            
            results.append(run_data)
            
        except MlflowException:
            print(f"Run ID {run_id} not found")
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def create_comparison_chart(
    runs_df: pd.DataFrame,
    metric_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Create a bar chart comparing runs on a specific metric
    
    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with runs comparison data
    metric_name : str
        Name of the metric to compare
    title : Optional[str]
        Chart title
    save_path : Optional[str]
        Path to save the chart
    
    Returns
    -------
    str
        Path to the saved chart
    """
    plt.figure(figsize=(10, 6))
    
    # Ensure all metric values are native Python types (not NumPy)
    chart_df = runs_df.copy()
    if metric_name in chart_df.columns:
        chart_df[metric_name] = chart_df[metric_name].apply(
            lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x
        )
    
    # Create chart using updated seaborn syntax
    ax = sns.barplot(
        x="run_name", 
        y=metric_name, 
        data=chart_df, 
        hue="run_name",  # Use hue instead of palette directly
        palette="viridis", 
        legend=False     # No need for legend since hue and x are the same
    )
    
    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):  # Check for NaN values
            ax.annotate(f"{height:.4f}", 
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', rotation=0)
    
    # Set title and labels
    plt.title(title or f"Comparison of {metric_name} Across Runs")
    plt.xlabel("Run")
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save chart
    if save_path is None:
        save_path = f"comparison_{metric_name}.png"
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path 
import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

class MLflowManager:
    """
    Manages MLflow experiment tracking and workflow management.
    """
    def __init__(self, experiment_name="qlib_workflow", tracking_uri=None):
        """
        Initialize MLflow tracking.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str, optional): URI for MLflow tracking server. If None, uses local storage.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use /tmp/mlruns to avoid /mnt/d path issues
            if not os.path.exists('mlruns'):
                os.system('mkdir -p /tmp/mlruns && ln -s /tmp/mlruns mlruns')
            
            tracking_uri = "file:mlruns"
            print(f"[DEBUG] Setting MLflow tracking URI to: {tracking_uri}")
            print(f"[DEBUG] MLflow directory symlinked to: {os.path.realpath('mlruns')}")
            mlflow.set_tracking_uri(tracking_uri)
            
        self.experiment_name = experiment_name
        self._setup_experiment()
        
    def _setup_experiment(self):
        """Set up the MLflow experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
    
    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        Args:
            run_name (str, optional): Name for the run. If None, uses timestamp.
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        print(f"[DEBUG] Started MLflow run in experiment {self.experiment_name}")
        print(f"[DEBUG] Run ID: {mlflow.active_run().info.run_id}")
        print(f"[DEBUG] Artifact URI: {mlflow.active_run().info.artifact_uri}")
        return mlflow.active_run().info.run_id
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def active_run(self):
        return mlflow.active_run()
    
    def log_parameters(self, params):
        """
        Log parameters to MLflow.
        
        Args:
            params (dict): Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
            
    def log_param(self, key, value):
        """
        Log a single parameter to MLflow.
        
        Args:
            key (str): Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number for the metrics
        """
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=step)
            except (ValueError, TypeError):
                print(f"Skipping metric {key} with non-numeric value: {value}")
    
    def log_artifact(self, local_path):
        """
        Log an artifact file to MLflow.
        
        Args:
            local_path (str): Path to the local file
        """
        artifact_abs_path = os.path.abspath(local_path)
        print(f"[DEBUG] Logging artifact from {artifact_abs_path}")
        print(f"[DEBUG] Current tracking URI: {mlflow.get_tracking_uri()}")
        print(f"[DEBUG] Current run artifact URI: {mlflow.active_run().info.artifact_uri}")
        mlflow.log_artifact(local_path)
    
    def log_model(self, model, artifact_path="model"):
        """
        Log a model to MLflow.
        
        Args:
            model: The model object to log
            artifact_path (str): Path within the artifact directory to store the model
        """
        mlflow.sklearn.log_model(model, artifact_path)
    
    def log_figure(self, figure, artifact_path):
        """
        Log a matplotlib figure to MLflow.
        
        Args:
            figure (matplotlib.figure.Figure): Figure to log
            artifact_path (str): Path to save the figure
        """
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        figure.savefig(temp_path)
        mlflow.log_artifact(temp_path, artifact_path)
        os.remove(temp_path)

def log_experiment(data, forecasts, backtest_results):
    """
    Log experiment results to MLflow.
    
    Args:
        data (pd.DataFrame): Historical stock data
        forecasts (dict): Forecasting results
        backtest_results (dict): Backtesting results
    """
    mlflow_manager = MLflowManager()
    
    # Start a new run
    run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_manager.start_run(run_name)
    
    try:
        # Log parameters
        params = {
            "symbols": ",".join(data.columns.get_level_values(1).unique()) if isinstance(data, pd.DataFrame) else "unknown",
            "forecast_horizon": len(forecasts) if isinstance(forecasts, list) else "unknown",
            "timestamp": datetime.now().isoformat()
        }
        mlflow_manager.log_parameters(params)
        
        # Log metrics from backtesting results
        if isinstance(backtest_results, dict):
            metrics = {}
            for symbol, results in backtest_results.items():
                for metric_name, metric_value in results.items():
                    metrics[f"{symbol}_{metric_name}"] = metric_value
            mlflow_manager.log_metrics(metrics)
        
        # Log data summary as artifact
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Save data description
            data_desc_path = "data_description.txt"
            with open(data_desc_path, "w") as f:
                f.write(str(data.describe()))
            mlflow_manager.log_artifact(data_desc_path)
            os.remove(data_desc_path)
            
            # Create and log a price chart
            if 'Close' in data.columns:
                plt.figure(figsize=(10, 6))
                data['Close'].plot()
                plt.title('Stock Close Prices')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.tight_layout()
                mlflow_manager.log_figure(plt.gcf(), "charts")
                plt.close()
        
        # Log forecasts as artifact
        forecast_path = "forecasts.json"
        with open(forecast_path, "w") as f:
            json.dump(forecasts, f, default=str)
        mlflow_manager.log_artifact(forecast_path)
        os.remove(forecast_path)
        
        print(f"Successfully logged experiment to MLflow. Run ID: {mlflow.active_run().info.run_id}")
    finally:
        # End the run
        mlflow_manager.end_run()
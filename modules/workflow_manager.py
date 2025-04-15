#!/usr/bin/env python3
"""
Workflow Manager for MarketSage quantitative analysis.
This module provides a workflow management system using MLflow for tracking
and event-driven architecture for module communication.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from modules.config_manager import get_config
from modules.logging_manager import get_logger
from modules.event_dispatcher import dispatch_event, EventTypes, EventPriority, add_event_handler, Event
from modules.mlflow_integration import MLflowManager
from modules.storage import StorageManager
from modules.indicators import calculate_all_indicators, RiskMetrics


# Configure logger
logger = get_logger(__name__)


class WorkflowStep:
    """Base class for workflow steps."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize workflow step.
        
        Args:
            name (str): Step name
            description (str): Step description
        """
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.result = None
        self.error = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow step.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Updated context
        """
        self.start_time = time.time()
        self.status = "running"
        
        # Dispatch event for step start
        dispatch_event(
            f"workflow.step.{self.name}.started",
            {
                "step_name": self.name,
                "step_description": self.description
            }
        )
        
        try:
            logger.info(f"Executing step: {self.name}")
            result = self._execute(context)
            self.status = "completed"
            self.result = result
            
            # Dispatch event for step completion
            dispatch_event(
                f"workflow.step.{self.name}.completed",
                {
                    "step_name": self.name,
                    "step_description": self.description,
                    "duration": time.time() - self.start_time
                }
            )
            
            return {**context, **result}
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"Step {self.name} failed: {e}")
            
            # Dispatch event for step failure
            dispatch_event(
                f"workflow.step.{self.name}.failed",
                {
                    "step_name": self.name,
                    "step_description": self.description,
                    "error": str(e)
                },
                EventPriority.HIGH
            )
            
            raise
        finally:
            self.end_time = time.time()
    
    def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement this method in subclasses.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Result to be added to context
        """
        raise NotImplementedError("Subclasses must implement _execute")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics for this step.
        
        Returns:
            Dict[str, float]: Metrics
        """
        metrics = {}
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            metrics[f"{self.name}_duration"] = duration
        
        return metrics
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for this step.
        
        Returns:
            Dict[str, Any]: Parameters
        """
        return {
            f"{self.name}_status": self.status
        }


class DataAcquisitionStep(WorkflowStep):
    """Step for acquiring data."""
    
    def __init__(self, data_manager, symbols: List[str], start_date: str, store: bool = True):
        """
        Initialize data acquisition step.
        
        Args:
            data_manager: Data acquisition manager
            symbols (List[str]): List of stock symbols
            start_date (str): Start date for historical data
            store (bool): Whether to store the fetched data
        """
        super().__init__(name="data_acquisition", description="Acquire historical data")
        self.data_manager = data_manager
        self.symbols = symbols
        self.start_date = start_date
        self.store = store
    
    def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data acquisition.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Result with acquired data
        """
        all_data = {}
        
        for symbol in self.symbols:
            logger.info(f"Fetching data for {symbol}...")
            data = self.data_manager.fetch_stock_data(
                symbol=symbol, 
                start_date=self.start_date,
                store=self.store
            )
            if not data.empty:
                all_data[symbol] = data
        
        # Calculate indicators if enabled
        calculate_indicators = get_config("workflow.calculate_indicators", False)
        if calculate_indicators:
            for symbol, data in all_data.items():
                logger.info(f"Calculating indicators for {symbol}...")
                all_data[symbol] = calculate_all_indicators(data)
        
        return {"data": all_data}
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for this step.
        
        Returns:
            Dict[str, Any]: Parameters
        """
        params = super().get_parameters()
        params.update({
            "symbols": ",".join(self.symbols),
            "start_date": self.start_date,
            "data_acquisition_symbols_count": len(self.symbols),
            "store_data": self.store
        })
        return params
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics for this step.
        
        Returns:
            Dict[str, float]: Metrics
        """
        metrics = super().get_metrics()
        
        if self.result and "data" in self.result:
            metrics["data_acquisition_symbols_success"] = len(self.result["data"])
        
        return metrics


class ForecastingStep(WorkflowStep):
    """Step for forecasting."""
    
    def __init__(self, forecasting_manager, model_type: str = "ensemble", optimize: bool = True):
        """
        Initialize forecasting step.
        
        Args:
            forecasting_manager: Forecasting manager
            model_type (str): Type of model to use
            optimize (bool): Whether to optimize model parameters
        """
        super().__init__(name="forecasting", description="Generate forecasts")
        self.forecasting_manager = forecasting_manager
        self.model_type = model_type
        self.optimize = optimize
    
    def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute forecasting.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Result with forecasts
        """
        if "data" not in context:
            raise ValueError("No data found in context. Data acquisition step must run first.")
        
        all_data = context["data"]
        forecasts = {}
        
        for symbol, data in all_data.items():
            logger.info(f"Forecasting {symbol}...")
            forecast_result = self.forecasting_manager.generate_forecast(
                symbol, 
                data, 
                model_type=self.model_type,
                use_stored_model=not self.optimize
            )
            if forecast_result:
                forecasts[symbol] = forecast_result
                
                # Dispatch forecasting event
                dispatch_event(EventTypes.FORECAST_COMPLETED, {
                    "symbol": symbol,
                    "forecast_horizon": forecast_result["forecast_horizon"],
                    "forecast_change_percent": forecast_result["forecast_change_percent"]
                })
        
        return {"forecasts": forecasts}
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics for this step.
        
        Returns:
            Dict[str, float]: Metrics
        """
        metrics = super().get_metrics()
        
        if self.result and "forecasts" in self.result:
            metrics["forecasting_symbols_count"] = len(self.result["forecasts"])
        
        return metrics


class BacktestingStep(WorkflowStep):
    """Step for backtesting."""
    
    def __init__(self, backtesting_manager, strategy: str = "momentum", optimize: bool = True):
        """
        Initialize backtesting step.
        
        Args:
            backtesting_manager: Backtesting manager
            strategy (str): Strategy to use for backtesting
            optimize (bool): Whether to optimize strategy parameters
        """
        super().__init__(name="backtesting", description="Run backtests")
        self.backtesting_manager = backtesting_manager
        self.strategy = strategy
        self.optimize = optimize
    
    def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute backtesting.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Result with backtest results
        """
        if "data" not in context:
            raise ValueError("No data found in context. Data acquisition step must run first.")
        
        if "forecasts" not in context:
            raise ValueError("No forecasts found in context. Forecasting step must run first.")
        
        all_data = context["data"]
        forecasts = context["forecasts"]
        bt_results = {}
        
        for symbol, data in all_data.items():
            if symbol in forecasts:
                logger.info(f"Backtesting {symbol}...")
                forecast_values = forecasts[symbol]["forecast_values"]
                
                if self.optimize:
                    # Optimize strategy parameters
                    logger.info(f"Optimizing strategy parameters for {symbol}...")
                    best_params, result = self.backtesting_manager.optimize_strategy(
                        symbol, data, forecast_values, self.strategy
                    )
                    bt_results[symbol] = result.get_metrics()
                else:
                    # Use default strategy
                    result = self.backtesting_manager.run_backtest(symbol, data, forecast_values, self.strategy)
                    bt_results[symbol] = result.get_metrics()
                
                # Dispatch backtesting event
                dispatch_event(EventTypes.BACKTEST_COMPLETED, {
                    "symbol": symbol,
                    "total_return": bt_results[symbol]["total_return"],
                    "sharpe_ratio": bt_results[symbol]["sharpe_ratio"]
                })
        
        return {"backtest_results": bt_results}
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics for this step.
        
        Returns:
            Dict[str, float]: Metrics
        """
        metrics = super().get_metrics()
        
        if self.result and "backtest_results" in self.result:
            bt_results = self.result["backtest_results"]
            for symbol, results in bt_results.items():
                for metric_name, metric_value in results.items():
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                        metrics[f"{symbol}_{metric_name}"] = metric_value
        
        return metrics


class VisualizationStep(WorkflowStep):
    """Step for creating visualizations."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualization step.
        
        Args:
            output_dir (str): Directory for output visualizations
        """
        super().__init__(name="visualization", description="Create visualizations")
        self.output_dir = output_dir
    
    def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute visualization creation.
        
        Args:
            context (Dict[str, Any]): The workflow context
            
        Returns:
            Dict[str, Any]: Result with visualization paths
        """
        if "data" not in context:
            raise ValueError("No data found in context. Data acquisition step must run first.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        visualization_paths = {}
        
        # Create price charts
        all_data = context["data"]
        for symbol, data in all_data.items():
            if not data.empty and 'Close' in data.columns:
                plt.figure(figsize=(10, 6))
                data['Close'].plot()
                plt.title(f'{symbol} Close Prices')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True)
                plt.tight_layout()
                
                # Save figure
                chart_path = os.path.join(self.output_dir, f"{symbol}_prices.png")
                plt.savefig(chart_path)
                plt.close()
                
                visualization_paths[f"{symbol}_price_chart"] = chart_path
        
        # Create forecast charts if forecasts exist
        if "forecasts" in context:
            forecasts = context["forecasts"]
            for symbol, forecast_result in forecasts.items():
                if symbol in all_data and "forecast_values" in forecast_result:
                    data = all_data[symbol]
                    forecast = forecast_result["forecast_values"]
                    
                    if not data.empty and 'Close' in data.columns:
                        plt.figure(figsize=(10, 6))
                        
                        # Plot historical data
                        plt.plot(data.index[-30:], data['Close'].iloc[-30:], label='Historical')
                        
                        # Plot forecast
                        last_date = data.index[-1]
                        forecast_dates = pd.date_range(
                            start=last_date, 
                            periods=len(forecast) + 1, 
                            freq='B'
                        )[1:]  # Business days, excluding the last historical date
                        
                        plt.plot(forecast_dates, forecast, 'r--', label='Forecast')
                        plt.title(f'{symbol} Price Forecast')
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        # Save figure
                        chart_path = os.path.join(self.output_dir, f"{symbol}_forecast.png")
                        plt.savefig(chart_path)
                        plt.close()
                        
                        visualization_paths[f"{symbol}_forecast_chart"] = chart_path
        
        return {"visualization_paths": visualization_paths}


class WorkflowManager:
    """
    Manages the execution of a workflow with MLflow tracking and event dispatching.
    """
    
    def __init__(
        self,
        experiment_name: str = None,
        tracking_uri: Optional[str] = None,
        artifact_dir: str = None,
        storage_backend: str = None
    ):
        """
        Initialize the workflow manager.
        
        Args:
            experiment_name (str, optional): Name of the MLflow experiment
            tracking_uri (str, optional): URI for MLflow tracking server
            artifact_dir (str, optional): Directory for artifacts
            storage_backend (str, optional): Storage backend to use
        """
        # Use configuration if parameters are not provided
        self.experiment_name = experiment_name or get_config("mlflow.experiment_name", "marketsage")
        self.tracking_uri = tracking_uri or get_config("mlflow.tracking_uri", None)
        self.artifact_dir = artifact_dir or get_config("mlflow.artifact_dir", "artifacts")
        self.storage_backend = storage_backend or get_config("storage.backend", "csv")
        
        # Initialize managers
        self.mlflow_manager = MLflowManager(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri
        )
        self.storage_manager = StorageManager(backend_type=self.storage_backend)
        
        # Initialize workflow
        self.steps = []
        self.context = {}
        
        # Create artifact directory
        os.makedirs(self.artifact_dir, exist_ok=True)
    
    def add_step(self, step: WorkflowStep) -> 'WorkflowManager':
        """
        Add a step to the workflow.
        
        Args:
            step (WorkflowStep): The workflow step to add
            
        Returns:
            WorkflowManager: Self for chaining
        """
        self.steps.append(step)
        return self
    
    def run(self) -> Dict[str, Any]:
        """
        Run the workflow with MLflow tracking.
        
        Returns:
            Dict[str, Any]: The final workflow context
        """
        run_name = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_manager.start_run(run_name)
        
        # Dispatch workflow start event
        dispatch_event(
            "workflow.started",
            {
                "run_name": run_name,
                "steps_count": len(self.steps),
                "steps": [step.name for step in self.steps]
            }
        )
        
        try:
            # Log workflow start
            self.mlflow_manager.log_parameters({
                "workflow_start_time": datetime.now().isoformat(),
                "steps_count": len(self.steps),
                "steps": ",".join(step.name for step in self.steps),
                "storage_backend": self.storage_backend
            })
            
            # Execute steps
            for step in self.steps:
                try:
                    self.context = step.execute(self.context)
                    
                    # Log step parameters and metrics
                    self.mlflow_manager.log_parameters(step.get_parameters())
                    self.mlflow_manager.log_metrics(step.get_metrics())
                    
                    # Log step artifacts
                    if step.name == "visualization" and "visualization_paths" in step.result:
                        for name, path in step.result["visualization_paths"].items():
                            self.mlflow_manager.log_artifact(path)
                    
                except Exception as e:
                    logger.error(f"Step {step.name} failed: {e}")
                    self.mlflow_manager.log_parameters({
                        f"{step.name}_error": str(e),
                        "workflow_status": "failed"
                    })
                    
                    # Dispatch workflow failure event
                    dispatch_event(
                        "workflow.failed",
                        {
                            "run_name": run_name,
                            "step_name": step.name,
                            "error": str(e)
                        },
                        EventPriority.HIGH
                    )
                    
                    raise
            
            # Log workflow completion
            self.mlflow_manager.log_parameters({
                "workflow_end_time": datetime.now().isoformat(),
                "workflow_status": "completed"
            })
            
            # Dispatch workflow completion event
            dispatch_event(
                "workflow.completed",
                {
                    "run_name": run_name,
                    "duration": time.time() - self.mlflow_manager.active_run().info.start_time / 1000
                }
            )
            
            return self.context
        
        finally:
            # Save context to artifact
            context_path = os.path.join(self.artifact_dir, "workflow_context.json")
            with open(context_path, "w") as f:
                # Convert non-serializable objects to strings
                serializable_context = {}
                for key, value in self.context.items():
                    if key == "data":
                        serializable_context[key] = {
                            symbol: "DataFrame" for symbol in value.keys()
                        }
                    elif isinstance(value, dict):
                        serializable_context[key] = {
                            k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
                            for k, v in value.items()
                        }
                    else:
                        serializable_context[key] = str(value)
                
                json.dump(serializable_context, f, indent=2, default=str)
            
            self.mlflow_manager.log_artifact(context_path)
            self.mlflow_manager.end_run()


# Create a standard workflow for backward compatibility
def create_workflow(
    data_acquisition_module,
    forecasting_module,
    backtesting_module,
    symbols,
    start_date,
    experiment_name="marketsage",
    tracking_uri=None
):
    """
    Create a standard workflow for quantitative analysis.
    
    Args:
        data_acquisition_module: Module for data acquisition
        forecasting_module: Module for forecasting
        backtesting_module: Module for backtesting
        symbols (List[str]): List of stock symbols
        start_date (str): Start date for historical data
        experiment_name (str): Name of the MLflow experiment
        tracking_uri (str, optional): URI for MLflow tracking server
        
    Returns:
        WorkflowManager: Configured workflow manager
    """
    workflow = WorkflowManager(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
    
    # Add steps
    workflow.add_step(
        DataAcquisitionStep(data_acquisition_module, symbols, start_date)
    ).add_step(
        ForecastingStep(forecasting_module)
    ).add_step(
        BacktestingStep(backtesting_module)
    ).add_step(
        VisualizationStep()
    )
    
    return workflow
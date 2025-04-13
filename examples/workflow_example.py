#!/usr/bin/env python3
"""
Example script demonstrating how to use the workflow manager with all new features.
This example shows how to use the configuration management, event dispatching,
logging, storage backends, indicators, and risk metrics.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from modules.config_manager import load_config_file, get_config
from modules.logging_manager import get_logger
from modules.event_dispatcher import dispatch_event, EventTypes, add_event_handler, get_event_dispatcher
from modules.scheduler import start_scheduler, stop_scheduler
from modules.storage import StorageManager
from modules.data_acquisition import DataAcquisitionManager
from modules.forecasting import ForecastingManager
from modules.backtesting import BacktestingManager
from modules.indicators import calculate_all_indicators, RiskMetrics
import modules.stock_manager as sm
from modules.workflow_manager import WorkflowManager, DataAcquisitionStep, ForecastingStep, BacktestingStep, VisualizationStep


# Configure logger
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Workflow Example with Advanced Features")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    
    # Data acquisition arguments
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of stock symbols or path to a file with symbols")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for historical data (YYYY-MM-DD)")
    
    # Storage arguments
    parser.add_argument("--storage-backend", type=str, default=None,
                        help="Storage backend to use (csv, sqlite, tsdb)")
    
    # MLflow arguments
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="URI for MLflow tracking server")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="MLflow experiment name")
    
    # Feature flags
    parser.add_argument("--calculate-indicators", action="store_true",
                        help="Calculate technical indicators")
    parser.add_argument("--optimize-models", action="store_true",
                        help="Optimize forecasting models")
    parser.add_argument("--optimize-strategies", action="store_true",
                        help="Optimize backtesting strategies")
    parser.add_argument("--run-scheduler", action="store_true",
                        help="Run the scheduler for periodic data updates")
    
    return parser.parse_args()


def get_symbols(symbols_arg):
    """Get list of symbols from argument (either comma-separated or file path)."""
    if not symbols_arg:
        symbols_file = get_config("data_acquisition.symbols_file", "stock_symbols.txt")
        if os.path.isfile(symbols_file):
            return sm.load_stock_symbols(symbols_file)
        else:
            return ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    if os.path.isfile(symbols_arg):
        return sm.load_stock_symbols(symbols_arg)
    else:
        return [symbol.strip() for symbol in symbols_arg.split(",")]


def setup_event_handlers():
    """Set up event handlers for workflow events."""
    # Log data acquisition events
    add_event_handler(
        lambda event: logger.info(f"Data acquired for {len(event.data.get('symbols', []))} symbols"),
        [EventTypes.DATA_ACQUIRED]
    )
    
    # Log forecasting events
    add_event_handler(
        lambda event: logger.info(f"Forecasting completed for {event.data.get('symbol')} with change {event.data.get('forecast_change_percent', 0):.2f}%"),
        [EventTypes.FORECAST_COMPLETED]
    )
    
    # Log backtesting events
    add_event_handler(
        lambda event: logger.info(f"Backtesting completed for {event.data.get('symbol')} with return {event.data.get('total_return', 0):.2f} and Sharpe {event.data.get('sharpe_ratio', 0):.2f}"),
        [EventTypes.BACKTEST_COMPLETED]
    )
    
    # Log workflow events
    add_event_handler(
        lambda event: logger.info(f"Workflow started with {event.data.get('steps_count', 0)} steps"),
        ["workflow.started"]
    )
    
    add_event_handler(
        lambda event: logger.info(f"Workflow completed in {event.data.get('duration', 0):.2f} seconds"),
        ["workflow.completed"]
    )
    
    add_event_handler(
        lambda event: logger.error(f"Workflow failed at step {event.data.get('step_name')}: {event.data.get('error')}"),
        ["workflow.failed"]
    )


def setup_scheduled_job(data_manager, symbols, start_date):
    """Set up a scheduled job for data updates."""
    from modules.scheduler import add_scheduled_job
    
    def update_data_job():
        """Job for updating stock data."""
        logger.info("Running scheduled data update job")
        
        for symbol in symbols:
            logger.info(f"Updating data for {symbol}...")
            data_manager.fetch_stock_data(symbol=symbol, start_date=start_date, store=True)
        
        logger.info(f"Data update completed for {len(symbols)} symbols")
    
    # Schedule job to run every minute for demonstration purposes
    # In a real application, you would use a more appropriate schedule
    # like "0 0 * * *" for daily at midnight
    add_scheduled_job("data_update", update_data_job, "* * * * *")
    logger.info("Scheduled data update job every minute")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    load_config_file(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override configuration with command line arguments
    if args.symbols:
        symbols_arg = args.symbols
    else:
        symbols_arg = get_config("data_acquisition.symbols_file", "stock_symbols.txt")
    
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = get_config("data_acquisition.start_date", "2020-01-01")
    
    if args.storage_backend:
        storage_backend = args.storage_backend
    else:
        storage_backend = get_config("storage.backend", "csv")
    
    if args.mlflow_tracking_uri:
        mlflow_tracking_uri = args.mlflow_tracking_uri
    else:
        mlflow_tracking_uri = get_config("mlflow.tracking_uri", None)
    
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = get_config("mlflow.experiment_name", "qlib_workflow_example")
    
    # Set up event handlers
    setup_event_handlers()
    
    # Start event dispatcher
    get_event_dispatcher().start()
    logger.info("Event dispatcher started")
    
    # Get symbols
    symbols = get_symbols(symbols_arg)
    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Initialize managers
    storage_manager = StorageManager(backend_type=storage_backend)
    data_manager = DataAcquisitionManager(storage_manager=storage_manager)
    forecasting_manager = ForecastingManager(storage_manager=storage_manager)
    backtesting_manager = BacktestingManager(storage_manager=storage_manager)
    
    # Start scheduler if requested
    if args.run_scheduler:
        start_scheduler()
        setup_scheduled_job(data_manager, symbols, start_date)
    
    # Create workflow
    workflow = WorkflowManager(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        storage_backend=storage_backend
    )
    
    # Add steps
    workflow.add_step(
        DataAcquisitionStep(data_manager, symbols, start_date, store=True)
    )
    
    # Add forecasting step with model optimization if requested
    workflow.add_step(
        ForecastingStep(forecasting_manager, model_type="ensemble", optimize=args.optimize_models)
    )
    
    # Add backtesting step with strategy optimization if requested
    workflow.add_step(
        BacktestingStep(backtesting_manager, strategy="momentum", optimize=args.optimize_strategies)
    )
    
    # Add visualization step
    workflow.add_step(
        VisualizationStep(output_dir="visualizations")
    )
    
    try:
        logger.info("Running workflow...")
        result = workflow.run()
        logger.info("Workflow completed successfully")
        
        # Print summary
        print("\nWorkflow Summary:")
        print(f"- Symbols processed: {len(symbols)}")
        print(f"- Storage backend: {storage_backend}")
        print(f"- Model optimization: {'Enabled' if args.optimize_models else 'Disabled'}")
        print(f"- Strategy optimization: {'Enabled' if args.optimize_strategies else 'Disabled'}")
        
        if "data" in result:
            print(f"- Data acquired for {len(result['data'])} symbols")
        
        if "forecasts" in result:
            print(f"- Forecasts generated for {len(result['forecasts'])} symbols")
            # Print forecast changes
            for symbol, forecast in result['forecasts'].items():
                print(f"  - {symbol}: {forecast['forecast_change_percent']:.2f}% change over {forecast['forecast_horizon']} days")
        
        if "backtest_results" in result:
            print(f"- Backtests completed for {len(result['backtest_results'])} symbols")
            # Print backtest returns
            for symbol, metrics in result['backtest_results'].items():
                print(f"  - {symbol}: {metrics['total_return']*100:.2f}% return, Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        
        if "visualization_paths" in result:
            print(f"- Visualizations created: {len(result['visualization_paths'])}")
            for name, path in result["visualization_paths"].items():
                print(f"  - {name}: {path}")
        
        print("\nView detailed results in the MLflow UI:")
        if mlflow_tracking_uri:
            print(f"- MLflow UI: {mlflow_tracking_uri}")
        else:
            print("- Start MLflow UI: mlflow ui")
        
        # Keep running if scheduler is active
        if args.run_scheduler:
            print("\nScheduler is running. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                stop_scheduler()
                get_event_dispatcher().stop()
    
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        
        # Stop scheduler and event dispatcher
        if args.run_scheduler:
            stop_scheduler()
        
        get_event_dispatcher().stop()
        sys.exit(1)
    
    # Stop event dispatcher if not running scheduler
    if not args.run_scheduler:
        get_event_dispatcher().stop()


if __name__ == "__main__":
    main()
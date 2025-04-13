#!/usr/bin/env python3
"""
Main entry point for the Quant Tool based on Microsoft Qlib.
This file orchestrates the data acquisition, forecasting, backtesting,
MLflow integration, and stock management components.
"""

import os
import sys
import json
import argparse
import pandas as pd
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import modules
from modules.config_manager import get_config, load_config_file
from modules.logging_manager import get_logger
from modules.event_dispatcher import dispatch_event, EventTypes, add_event_handler, get_event_dispatcher
from modules.scheduler import start_scheduler, stop_scheduler
from modules.storage import StorageManager
from modules.data_acquisition import DataAcquisitionManager
from modules.forecasting import ForecastingManager
from modules.backtesting import BacktestingManager
from modules.mlflow_integration import MLflowManager
from modules.indicators import calculate_all_indicators, RiskMetrics
import modules.stock_manager as sm


# Configure logger
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quant Tool based on Microsoft Qlib")
    
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
    
    # Workflow control
    parser.add_argument("--skip-forecasting", action="store_true",
                        help="Skip the forecasting step")
    parser.add_argument("--skip-backtesting", action="store_true",
                        help="Skip the backtesting step")
    parser.add_argument("--skip-mlflow", action="store_true",
                        help="Skip MLflow logging")
    parser.add_argument("--run-scheduler", action="store_true",
                        help="Run the scheduler for periodic data updates")
    
    # Advanced options
    parser.add_argument("--indicators", action="store_true",
                        help="Calculate technical indicators")
    parser.add_argument("--risk-metrics", action="store_true",
                        help="Calculate risk metrics")
    
    return parser.parse_args()


def get_symbols(symbols_arg):
    """Get list of symbols from argument (either comma-separated or file path)."""
    if not symbols_arg:
        return []
        
    if os.path.isfile(symbols_arg):
        return sm.load_stock_symbols(symbols_arg)
    else:
        return [symbol.strip() for symbol in symbols_arg.split(",")]


def setup_event_handlers():
    """Set up event handlers."""
    # Log data acquisition events
    add_event_handler(
        lambda event: logger.info(f"Data acquired for {len(event.data.get('symbols', []))} symbols"),
        [EventTypes.DATA_ACQUIRED]
    )
    
    # Log forecasting events
    add_event_handler(
        lambda event: logger.info(f"Forecasting completed for {event.data.get('symbol')}"),
        [EventTypes.FORECAST_COMPLETED]
    )
    
    # Log backtesting events
    add_event_handler(
        lambda event: logger.info(f"Backtesting completed for {event.data.get('symbol')}"),
        [EventTypes.BACKTEST_COMPLETED]
    )
    
    # Log error events
    add_event_handler(
        lambda event: logger.error(f"Error occurred: {event.data.get('message')}"),
        [EventTypes.ERROR_OCCURRED]
    )


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received, cleaning up...")
        stop_scheduler()
        get_event_dispatcher().stop()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on exit
    atexit.register(lambda: stop_scheduler())
    atexit.register(lambda: get_event_dispatcher().stop())


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load configuration
    load_config_file(args.config)
    
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
        experiment_name = get_config("mlflow.experiment_name", "qlib_workflow")
    
    # Set up event handlers
    setup_event_handlers()
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Start event dispatcher
    get_event_dispatcher().start()
    
    # Start scheduler if requested
    if args.run_scheduler:
        start_scheduler()
        logger.info("Scheduler started for periodic data updates")
    
    logger.info("Starting Quant Tool Pipeline...")
    
    # Get symbols
    symbols = get_symbols(symbols_arg)
    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Initialize managers
    storage_manager = StorageManager(backend_type=storage_backend)
    data_manager = DataAcquisitionManager(storage_manager=storage_manager)
    forecasting_manager = ForecastingManager(storage_manager=storage_manager)
    backtesting_manager = BacktestingManager(storage_manager=storage_manager)
    
    # Data Acquisition
    logger.info("Acquiring historical data...")
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        data = data_manager.fetch_stock_data(symbol=symbol, start_date=start_date, store=True)
        if not data.empty:
            all_data[symbol] = data
            
            # Calculate indicators if requested
            if args.indicators:
                logger.info(f"Calculating indicators for {symbol}...")
                all_data[symbol] = calculate_all_indicators(data)
    
    if not all_data:
        logger.warning("No data acquired. Exiting.")
        return
    
    logger.info(f"Data acquired for {len(all_data)} symbols.")
    
    # Dispatch data acquisition event
    dispatch_event(EventTypes.DATA_ACQUIRED, {"symbols": list(all_data.keys())})
    
    # Calculate risk metrics if requested
    if args.risk_metrics:
        logger.info("Calculating risk metrics...")
        risk_metrics = {}
        for symbol, data in all_data.items():
            returns = data['Close'].pct_change().dropna()
            
            risk_metrics[symbol] = {
                "var_95": RiskMetrics.calculate_var(returns, 0.95),
                "cvar_95": RiskMetrics.calculate_cvar(returns, 0.95),
                "max_drawdown": RiskMetrics.calculate_max_drawdown(data['Close']),
                "sharpe_ratio": RiskMetrics.calculate_sharpe_ratio(returns),
                "sortino_ratio": RiskMetrics.calculate_sortino_ratio(returns)
            }
            
            logger.info(f"Risk metrics for {symbol}: VaR={risk_metrics[symbol]['var_95']:.4f}, "
                      f"Max Drawdown={risk_metrics[symbol]['max_drawdown']:.4f}%")
    
    # Forecasting
    forecasts = {}
    if not args.skip_forecasting:
        logger.info("Running forecasting...")
        for symbol, data in all_data.items():
            logger.info(f"Forecasting {symbol}...")
            forecast_result = forecasting_manager.generate_forecast(symbol, data)
            if forecast_result:
                forecasts[symbol] = forecast_result
                
                # Print detailed forecast information
                logger.info(f"Forecast for {symbol}:")
                logger.info(f"  Last price: ${forecast_result['last_price']:.2f}")
                logger.info(f"  2-week forecast: ${forecast_result['forecast_values'][-1]:.2f} ({forecast_result['forecast_change_percent']:.2f}%)")
                logger.info(f"  Volatility: {forecast_result['volatility']:.2f}%")
                logger.info(f"  Trend: {forecast_result['trend']:.2f}%")
                logger.info(f"  Overall recommendation: {forecast_result['overall_action']}")
                
                # Daily forecasts
                logger.info("  Daily forecasts:")
                for i, (price, change, action) in enumerate(zip(
                    forecast_result['forecast_values'],
                    forecast_result['daily_changes'],
                    forecast_result['daily_actions']
                )):
                    logger.info(f"    Day {i+1}: ${price:.2f} ({change:+.2f}%) - {action}")
                
                # Dispatch forecasting event
                dispatch_event(EventTypes.FORECAST_COMPLETED, {
                    "symbol": symbol,
                    "forecast_horizon": forecast_result["forecast_horizon"],
                    "forecast_change_percent": forecast_result["forecast_change_percent"],
                    "overall_action": forecast_result["overall_action"]
                })
        
        logger.info(f"Forecasting completed for {len(forecasts)} symbols.")
    else:
        logger.info("Forecasting step skipped.")
    
    # Backtesting
    bt_results = {}
    optimization_results = {}
    if not args.skip_backtesting and forecasts:
        logger.info("Running backtesting and strategy optimization...")
        for symbol, data in all_data.items():
            if symbol in forecasts:
                logger.info(f"Backtesting {symbol}...")
                forecast_values = forecasts[symbol]["forecast_values"]
                
                # Run basic backtest with default strategy
                result = backtesting_manager.run_backtest(symbol, data, forecast_values)
                bt_results[symbol] = result.get_metrics()
                
                # Optimize strategy parameters
                logger.info(f"Optimizing strategy parameters for {symbol}...")
                
                # Try different strategies and find the best one
                best_strategy = None
                best_params = None
                best_result = None
                best_sharpe = -float('inf')
                best_stats = None
                
                for strategy_name in ['momentum', 'mean_reversion', 'forecast_based']:
                    logger.info(f"  Testing {strategy_name} strategy...")
                    params, result, stats = backtesting_manager.optimize_strategy(
                        symbol, data, forecast_values, strategy_name=strategy_name
                    )
                    
                    metrics = result.get_metrics()
                    if metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_strategy = strategy_name
                        best_params = params
                        best_result = result
                        best_stats = stats
                
                # Store optimization results
                optimization_results[symbol] = {
                    'strategy': best_strategy,
                    'params': best_params,
                    'metrics': best_result.get_metrics(),
                    'stats': best_stats
                }
                
                # Log optimization results
                logger.info(f"Optimization results for {symbol}:")
                logger.info(f"  Best strategy: {best_strategy}")
                logger.info(f"  Best parameters: {best_params}")
                logger.info(f"  Sharpe ratio: {best_result.sharpe_ratio:.4f}")
                logger.info(f"  Annual return: {best_result.annual_return:.2%}")
                logger.info(f"  Max drawdown: {best_result.max_drawdown:.2%}")
                
                # Update backtest results with optimized results
                bt_results[symbol] = best_result.get_metrics()
                
                # Dispatch backtesting event
                dispatch_event(EventTypes.BACKTEST_COMPLETED, {
                    "symbol": symbol,
                    "strategy": best_strategy,
                    "total_return": bt_results[symbol]["total_return"],
                    "sharpe_ratio": bt_results[symbol]["sharpe_ratio"],
                    "annual_return": bt_results[symbol]["annual_return"]
                })
        
        logger.info(f"Backtesting completed with results for {len(bt_results)} symbols.")
    else:
        logger.info("Backtesting step skipped.")
    
    # MLflow Integration
    if not args.skip_mlflow:
        logger.info("Logging performance metrics to MLflow...")
        # Initialize MLflow with tracking URI if provided
        mlflow_manager = MLflowManager(
            experiment_name=experiment_name,
            tracking_uri=mlflow_tracking_uri
        )
        
        # Start a new run
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_manager.start_run(run_name)
        
        try:
            # Log parameters
            params = {
                "symbols": ",".join(symbols),
                "start_date": start_date,
                "storage_backend": storage_backend,
                "forecast_enabled": not args.skip_forecasting,
                "backtest_enabled": not args.skip_backtesting,
                "indicators_enabled": args.indicators,
                "risk_metrics_enabled": args.risk_metrics
            }
            mlflow_manager.log_parameters(params)
            
            # Log metrics from backtesting results
            if bt_results:
                metrics = {}
                for symbol, results in bt_results.items():
                    for metric_name, metric_value in results.items():
                        metrics[f"{symbol}_{metric_name}"] = metric_value
                mlflow_manager.log_metrics(metrics)
                
                # Log optimization results
                if optimization_results:
                    for symbol, opt_result in optimization_results.items():
                        # Log best strategy and parameters
                        mlflow_manager.log_param(f"{symbol}_best_strategy", opt_result['strategy'])
                        mlflow_manager.log_param(f"{symbol}_best_params", str(opt_result['params']))
                        
                        # Log parameter impact analysis
                        if 'stats' in opt_result and 'parameter_impact' in opt_result['stats']:
                            impact_path = f"{symbol}_parameter_impact.json"
                            with open(impact_path, "w") as f:
                                json.dump(opt_result['stats']['parameter_impact'], f, indent=2)
                            mlflow_manager.log_artifact(impact_path)
                            os.remove(impact_path)
            
            # Log risk metrics if calculated
            if args.risk_metrics and risk_metrics:
                for symbol, metrics_dict in risk_metrics.items():
                    for metric_name, metric_value in metrics_dict.items():
                        mlflow_manager.log_metric(f"{symbol}_{metric_name}", metric_value)
            
            # Log data summary
            for symbol, data in all_data.items():
                if not data.empty:
                    # Save data description
                    data_desc_path = f"{symbol}_data_description.txt"
                    with open(data_desc_path, "w") as f:
                        f.write(str(data.describe()))
                    mlflow_manager.log_artifact(data_desc_path)
                    os.remove(data_desc_path)
            
            # Log forecasts
            if forecasts:
                # Log detailed forecast information
                forecast_path = "forecasts.json"
                # Create a more detailed forecast object for logging
                detailed_forecasts = {}
                for symbol, forecast_data in forecasts.items():
                    detailed_forecasts[symbol] = {
                        "forecast_values": forecast_data["forecast_values"],
                        "daily_changes": forecast_data["daily_changes"],
                        "daily_actions": forecast_data["daily_actions"],
                        "volatility": forecast_data["volatility"],
                        "trend": forecast_data["trend"],
                        "overall_action": forecast_data["overall_action"]
                    }
                with open(forecast_path, "w") as outfile:
                    json.dump(detailed_forecasts, outfile, default=str)
                mlflow_manager.log_artifact(forecast_path)
                os.remove(forecast_path)
                
                # Create and log forecast charts
                try:
                    import matplotlib.pyplot as plt
                    for symbol, forecast_data in forecasts.items():
                        if symbol in all_data:
                            # Create forecast chart
                            plt.figure(figsize=(10, 6))
                            
                            # Plot historical data
                            hist_dates = all_data[symbol].index[-30:]  # Last 30 days
                            hist_prices = all_data[symbol]['Close'].iloc[-30:]
                            plt.plot(range(len(hist_dates)), hist_prices, 'b-', label='Historical')
                            
                            # Plot forecast
                            forecast_dates = range(len(hist_dates), len(hist_dates) + len(forecast_data['forecast_values']))
                            plt.plot(forecast_dates, forecast_data['forecast_values'], 'r--', label='Forecast')
                            
                            # Add labels and legend
                            plt.title(f'{symbol} Price Forecast')
                            plt.xlabel('Trading Days')
                            plt.ylabel('Price ($)')
                            plt.legend()
                            plt.grid(True)
                            
                            # Save and log
                            chart_path = f"{symbol}_forecast_chart.png"
                            plt.savefig(chart_path)
                            mlflow_manager.log_artifact(chart_path)
                            os.remove(chart_path)
                            plt.close()
                except Exception as e:
                    logger.error(f"Error creating forecast charts: {e}")
            
            logger.info(f"Successfully logged to MLflow. Run ID: {mlflow_manager.active_run().info.run_id}")
        finally:
            # End the run
            mlflow_manager.end_run()
    else:
        logger.info("MLflow logging step skipped.")
    
    # Stock Management
    logger.info("Managing stocks...")
    for symbol, data in all_data.items():
        sm.manage_stocks(data)
    
    logger.info("Quant Tool Pipeline completed.")
    
    # Keep running if scheduler is active
    if args.run_scheduler:
        logger.info("Scheduler is running. Press Ctrl+C to exit.")
        try:
            # Keep the main thread alive
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting...")


if __name__ == "__main__":
    main()
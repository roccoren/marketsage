#!/usr/bin/env python3
"""
StockScan Backtest Example

This script demonstrates how to use the StockScan library
to run a backtest using the LEAN framework.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.StockScan import (
    create_engine, 
    create_data_handler, 
    LEAN_AVAILABLE,
    get_available_algorithms
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_results(results):
    """
    Display backtest results.
    
    Args:
        results: Dictionary containing backtest results
    """
    print("\n===== Backtest Results =====")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    
    metrics = results['metrics']
    print("\n----- Performance Metrics -----")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Show result files if available
    if results['result_files']:
        print("\n----- Result Files -----")
        for name, path in results['result_files'].items():
            print(f"- {name}: {path}")


async def prepare_data():
    """
    Prepare data for backtesting by fetching and converting to LEAN format.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data handler
        data_handler = create_data_handler()
        
        # Define data parameters
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        # Fetch data for each symbol
        print(f"Fetching data for {symbols}...")
        data_dict = await data_handler.GetMultipleSymbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            resolution="Daily"
        )
        
        # Convert data to LEAN format
        print("Converting data to LEAN format...")
        for symbol, data in data_dict.items():
            if not data.empty:
                output_path = data_handler.ConvertToLeanFormat(data, symbol)
                print(f"Converted data for {symbol} saved to {output_path}")
            else:
                print(f"No data available for {symbol}")
        
        return True
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return False


def run_trend_following_backtest():
    """Run a backtest using the TrendFollowingAlgorithm."""
    if not LEAN_AVAILABLE:
        print("LEAN framework is not available.")
        return
    
    try:
        # Create backtesting engine
        engine = create_engine()
        
        # Print available algorithms
        print("Available algorithms:", get_available_algorithms())
        
        # Define backtest parameters
        symbols = ["AAPL", "MSFT"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Algorithm-specific parameters
        parameters = {
            "FastPeriod": 10,
            "SlowPeriod": 30,
            "VolumeThreshold": 1.2,
            "EnableShort": True,
            "UseATRForStops": True,
            "ATRMultiplier": 2.0
        }
        
        # Run backtest
        print(f"Running TrendFollowingAlgorithm backtest for {symbols}...")
        results = engine.RunBacktest(
            algorithm_name="TrendFollowingAlgorithm",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            resolution="Daily",
            parameters=parameters
        )
        
        # Display results
        display_results(results)
        
        # Analyze results
        analysis = engine.AnalyzeResults(results)
        print("\n----- Analysis Results -----")
        for category, metrics in analysis.items():
            print(f"\n{category.title()}:")
            for name, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: {value}")
        
        return results
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return None


def run_mean_reversion_backtest():
    """Run a backtest using the MeanReversionAlgorithm."""
    if not LEAN_AVAILABLE:
        print("LEAN framework is not available.")
        return
    
    try:
        # Create backtesting engine
        engine = create_engine()
        
        # Define backtest parameters
        symbols = ["AAPL", "MSFT"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Algorithm-specific parameters
        parameters = {
            "RsiPeriod": 14,
            "OversoldThreshold": 30,
            "OverboughtThreshold": 70,
            "UseBollingerConfirmation": True,
            "EnableShort": True
        }
        
        # Run backtest
        print(f"Running MeanReversionAlgorithm backtest for {symbols}...")
        results = engine.RunBacktest(
            algorithm_name="MeanReversionAlgorithm",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            resolution="Daily",
            parameters=parameters
        )
        
        # Display results
        display_results(results)
        
        return results
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return None


def optimize_trend_following():
    """Run parameter optimization for the TrendFollowingAlgorithm."""
    if not LEAN_AVAILABLE:
        print("LEAN framework is not available.")
        return
    
    try:
        # Create backtesting engine
        engine = create_engine()
        
        # Define backtest parameters
        symbols = ["AAPL"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Parameter space to search
        parameter_space = {
            "FastPeriod": [5, 10, 15, 20],
            "SlowPeriod": [20, 30, 40, 50],
            "EnableShort": [False, True]
        }
        
        # Run optimization
        print(f"Optimizing TrendFollowingAlgorithm for {symbols}...")
        results = engine.OptimizeParameters(
            algorithm_name="TrendFollowingAlgorithm",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            parameter_space=parameter_space,
            initial_capital=100000,
            resolution="Daily",
            optimization_metric="SharpeRatio"
        )
        
        # Display best results
        best_result = results['best_result']
        print("\n===== Best Parameters =====")
        for param, value in best_result['parameters'].items():
            print(f"{param}: {value}")
        
        print("\n===== Best Result Performance =====")
        display_results(best_result)
        
        return results
    except Exception as e:
        logger.error(f"Error optimizing algorithm: {e}")
        return None


def run_all_backtests():
    """Run all example backtests."""
    # Prepare data
    asyncio.run(prepare_data())
    
    # Run trend following backtest
    print("\n\n===============================================")
    print("Running Trend Following Algorithm Backtest")
    print("===============================================")
    run_trend_following_backtest()
    
    # Run mean reversion backtest
    print("\n\n===============================================")
    print("Running Mean Reversion Algorithm Backtest")
    print("===============================================")
    run_mean_reversion_backtest()
    
    # Run parameter optimization
    print("\n\n===============================================")
    print("Running Parameter Optimization")
    print("===============================================")
    optimize_trend_following()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="StockScan Backtest Example")
    parser.add_argument('--algorithm', choices=['trend', 'mean_reversion', 'optimize', 'all'], 
                       default='all', help='Algorithm to run')
    args = parser.parse_args()
    
    # Check if LEAN is available
    if not LEAN_AVAILABLE:
        print("LEAN framework is not available. Please install the required dependencies.")
        print("pip install quantconnect-lean quantconnect-stubs pythonnet")
        sys.exit(1)
    
    # Prepare data
    print("Preparing data for backtesting...")
    asyncio.run(prepare_data())
    
    # Run selected backtest
    if args.algorithm == 'trend':
        run_trend_following_backtest()
    elif args.algorithm == 'mean_reversion':
        run_mean_reversion_backtest()
    elif args.algorithm == 'optimize':
        optimize_trend_following()
    else:  # 'all'
        run_all_backtests() 
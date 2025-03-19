#!/usr/bin/env python3
"""
LEAN Backtest Example

This example script demonstrates how to use the LEAN framework to run backtests
programmatically, without using the command-line interface.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the LEAN adapter
try:
    from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode
except ImportError:
    print("ERROR: LEAN adapter not found. Make sure the LEAN framework is installed.")
    print("See requirements.txt for LEAN-related packages.")
    sys.exit(1)


def run_trend_following_backtest():
    """Run a backtest using the TrendFollowingAlgorithm."""
    print("Running TrendFollowing backtest...")
    
    # Create adapter
    adapter = LeanAdapter(
        config_path="config/lean_config.json",
        mode=BacktestMode.LEAN,
        data_folder="data",
        results_folder="results/lean_backtest"
    )
    
    # Set up backtest parameters
    symbols = ["SPY", "AAPL", "MSFT", "GOOG"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    # Algorithm-specific parameters
    parameters = {
        "fast_period": 10,
        "slow_period": 30,
        "signal_threshold": 0.0,
        "volume_threshold": 1.5,
        "position_size": 0.1
    }
    
    # Run backtest
    results = adapter.run_backtest(
        algorithm_name="TrendFollowingAlgorithm",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        resolution="Daily",
        parameters=parameters
    )
    
    # Print results
    print("\nTrendFollowing Backtest Results:")
    print(f"Algorithm: {results.get('results', {}).get('algorithm_name', 'Unknown')}")
    
    if 'metrics' in results.get('results', {}):
        metrics = results['results']['metrics']
        print(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
    
    return results


def run_mean_reversion_backtest():
    """Run a backtest using the MeanReversionAlgorithm."""
    print("Running MeanReversion backtest...")
    
    # Create adapter
    adapter = LeanAdapter(
        config_path="config/lean_config.json",
        mode=BacktestMode.LEAN,
        data_folder="data",
        results_folder="results/lean_backtest"
    )
    
    # Set up backtest parameters
    symbols = ["SPY", "QQQ", "IWM"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    # Algorithm-specific parameters
    parameters = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.10,
        "position_size": 0.1
    }
    
    # Run backtest
    results = adapter.run_backtest(
        algorithm_name="MeanReversionAlgorithm",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        resolution="Daily",
        parameters=parameters
    )
    
    # Print results
    print("\nMeanReversion Backtest Results:")
    print(f"Algorithm: {results.get('results', {}).get('algorithm_name', 'Unknown')}")
    
    if 'metrics' in results.get('results', {}):
        metrics = results['results']['metrics']
        print(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
    
    return results


def run_all_backtests():
    """Run all backtests and compare results."""
    # Run both backtests
    tf_results = run_trend_following_backtest()
    mr_results = run_mean_reversion_backtest()
    
    # Compare results
    print("\nBacktest Comparison:")
    
    tf_metrics = tf_results.get('results', {}).get('metrics', {})
    mr_metrics = mr_results.get('results', {}).get('metrics', {})
    
    print(f"{'Metric':<20} {'TrendFollowing':<15} {'MeanReversion':<15}")
    print("-" * 50)
    
    # Total Return
    tf_return = tf_metrics.get('total_return', 0.0)
    mr_return = mr_metrics.get('total_return', 0.0)
    print(f"{'Total Return':<20} {tf_return:.2%:<15} {mr_return:.2%:<15}")
    
    # Sharpe Ratio
    tf_sharpe = tf_metrics.get('sharpe_ratio', 0.0)
    mr_sharpe = mr_metrics.get('sharpe_ratio', 0.0)
    print(f"{'Sharpe Ratio':<20} {tf_sharpe:.2f:<15} {mr_sharpe:.2f:<15}")
    
    # Max Drawdown
    tf_dd = tf_metrics.get('max_drawdown', 0.0)
    mr_dd = mr_metrics.get('max_drawdown', 0.0)
    print(f"{'Max Drawdown':<20} {tf_dd:.2%:<15} {mr_dd:.2%:<15}")
    
    # Win Rate
    tf_wr = tf_metrics.get('win_rate', 0.0)
    mr_wr = mr_metrics.get('win_rate', 0.0)
    if tf_wr is not None and mr_wr is not None:
        print(f"{'Win Rate':<20} {tf_wr:.2%:<15} {mr_wr:.2%:<15}")


def list_available_algorithms():
    """List all available LEAN algorithms."""
    adapter = LeanAdapter()
    algorithms = adapter.get_available_algorithms()
    
    print("\nAvailable LEAN Algorithms:")
    for algorithm in algorithms:
        print(f"- {algorithm}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LEAN Backtest Example")
    parser.add_argument('--algorithm', choices=['trend', 'mean_reversion', 'all', 'list'], 
                        default='all', help='Algorithm to run')
    
    args = parser.parse_args()
    
    if args.algorithm == 'trend':
        run_trend_following_backtest()
    elif args.algorithm == 'mean_reversion':
        run_mean_reversion_backtest()
    elif args.algorithm == 'all':
        run_all_backtests()
    elif args.algorithm == 'list':
        list_available_algorithms()
    else:
        print(f"Unknown algorithm: {args.algorithm}") 
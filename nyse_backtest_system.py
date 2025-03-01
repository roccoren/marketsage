#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Import project modules
from src.utils.exchange_data_loader import ExchangeDataLoader
from src.strategies.multi_indicator_strategy import MultiIndicatorStrategy
from src.strategies.backtester import Backtester
from src.strategies.optimizer import StrategyOptimizer
from src.utils.technical_indicators import TechnicalIndicators


def load_nyse_data(start_date: str = '2020-01-01') -> dict:
    """
    Load NYSE stock data from the data/NYSE folder
    
    Args:
        start_date: Start date for the data in format 'YYYY-MM-DD'
        
    Returns:
        Dictionary of DataFrames with symbol keys
    """
    logger.info(f"Loading NYSE stock data from {start_date} to present...")
    
    data_loader = ExchangeDataLoader()
    nyse_dir = Path('data/NYSE')
    
    if not nyse_dir.exists():
        logger.error(f"Directory {nyse_dir} does not exist")
        return {}
    
    stock_data = {}
    symbol_files = list(nyse_dir.glob('*.json'))
    
    logger.info(f"Found {len(symbol_files)} NYSE symbols")
    
    # Load data for each symbol
    for symbol_file in symbol_files:
        symbol = symbol_file.stem
        try:
            data = data_loader.load_stock_data('NYSE', symbol)
            
            if data is not None and len(data) > 0:
                # Filter data from start_date
                start_datetime = pd.Timestamp(start_date)
                data = data[data.index >= start_datetime]
                
                if len(data) > 50:  # Ensure sufficient data points
                    stock_data[symbol] = data
                    logger.debug(f"Loaded {len(data)} data points for {symbol}")
            else:
                logger.debug(f"No data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    logger.info(f"Successfully loaded data for {len(stock_data)} NYSE symbols")
    return stock_data


def optimize_strategy(training_data: pd.DataFrame, param_grid: Dict = None) -> Dict:
    """
    Optimize strategy parameters using historical data
    
    Args:
        training_data: Historical price data for optimization
        param_grid: Parameter grid for optimization
        
    Returns:
        Dictionary with optimized parameters
    """
    logger.info("Optimizing strategy parameters...")
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            # RSI parameters
            'rsi_period': [7, 14, 21],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            
            # MACD parameters
            'macd_fast': [8, 12, 16],
            'macd_slow': [21, 26, 30],
            'macd_signal': [7, 9, 11],
            
            # Bollinger Bands parameters
            'bb_period': [15, 20, 25],
            'bb_std': [1.8, 2.0, 2.2],
            
            # Strategy weights (fixed to ensure they sum to 1.0)
            'trend_weight': [0.35],
            'momentum_weight': [0.3],
            'volatility_weight': [0.15],
            'macd_weight': [0.2],
            
            # Risk parameters
            'atr_multiplier': [2.0, 2.5, 3.0]
        }
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(
        strategy_class=MultiIndicatorStrategy,
        param_grid=param_grid,
        data=training_data,
        train_size=0.8,
        metric='sharpe_ratio'
    )
    
    # Run optimization
    results = optimizer.optimize(initial_capital=100000)
    
    if results['best_params'] is None:
        logger.error("Optimization failed to find valid parameters")
        return {}
        
    logger.info(f"Optimization complete. Best parameters: {results['best_params']}")
    return results['best_params']


def run_parallel_backtest(stock_data: dict, params: dict) -> dict:
    """
    Run backtests in parallel across multiple stocks
    
    Args:
        stock_data: Dictionary of stock data (symbol -> DataFrame)
        params: Strategy parameters
        
    Returns:
        Dictionary of backtest results (symbol -> metrics)
    """
    logger.info(f"Running parallel backtests on {len(stock_data)} stocks...")
    
    strategy = MultiIndicatorStrategy(**params)
    results = {}
    
    # Process stocks in parallel
    with ProcessPoolExecutor() as executor:
        futures = {
            symbol: executor.submit(run_single_backtest, data, strategy)
            for symbol, data in stock_data.items()
        }
        
        # Collect results
        for symbol, future in futures.items():
            try:
                results[symbol] = future.result()
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {str(e)}")
                
    return results


def run_single_backtest(data: pd.DataFrame, strategy) -> dict:
    """
    Run backtest for a single stock
    
    Args:
        data: OHLCV DataFrame with datetime index
        strategy: Strategy instance
        
    Returns:
        Dictionary of performance metrics
    """
    backtester = Backtester(strategy)
    result = backtester.run(data)
    return result.metrics


def analyze_recent_signals(stock_data: dict, strategy_params: dict, days: int = 7) -> dict:
    """
    Analyze recent trading signals for the last N days
    
    Args:
        stock_data: Dictionary of stock data (symbol -> DataFrame)
        strategy_params: Strategy parameters
        days: Number of recent days to analyze
        
    Returns:
        Dictionary of recent signals (symbol -> signals)
    """
    logger.info(f"Analyzing recent signals for the last {days} days...")
    
    strategy = MultiIndicatorStrategy(**strategy_params)
    recent_signals = {}
    
    # Calculate end date (last 7 trading days)
    end_date = datetime.now()
    
    for symbol, data in stock_data.items():
        # Make sure we have recent data
        if data.index[-1] < (end_date - timedelta(days=30)):
            logger.debug(f"Skipping {symbol} - data not recent enough")
            continue
            
        # Generate signals for recent days
        recent_data = data.iloc[-days*2:]  # Get extra history for calculations
        if len(recent_data) < 30:  # Need enough data for indicators
            continue
            
        signals = strategy.analyze(recent_data)
        
        # Filter signals from the last N days
        cutoff_date = end_date - timedelta(days=days)
        recent_signals_list = [
            signal for signal in signals 
            if 'Date' in signal and signal['Date'] >= cutoff_date
        ]
        
        if recent_signals_list:
            recent_signals[symbol] = recent_signals_list
            
    return recent_signals


def calculate_performance_summary(results: dict) -> dict:
    """
    Calculate summary statistics across all backtest results
    
    Args:
        results: Dictionary of backtest results (symbol -> metrics)
        
    Returns:
        Dictionary of summary statistics
    """
    if not results:
        return {}
        
    # Extract metrics
    all_metrics = {
        'sharpe_ratio': [res['sharpe_ratio'] for res in results.values() if 'sharpe_ratio' in res],
        'sortino_ratio': [res['sortino_ratio'] for res in results.values() if 'sortino_ratio' in res],
        'win_rate': [res['win_rate'] for res in results.values() if 'win_rate' in res],
        'profit_factor': [res['profit_factor'] for res in results.values() if 'profit_factor' in res],
        'max_drawdown': [res['max_drawdown'] for res in results.values() if 'max_drawdown' in res],
        'total_pnl': [res['total_pnl'] for res in results.values() if 'total_pnl' in res],
        'total_trades': [res['total_trades'] for res in results.values() if 'total_trades' in res]
    }
    
    # Calculate averages
    summary = {
        'avg_sharpe_ratio': sum(all_metrics['sharpe_ratio']) / len(all_metrics['sharpe_ratio']) if all_metrics['sharpe_ratio'] else 0,
        'avg_sortino_ratio': sum(all_metrics['sortino_ratio']) / len(all_metrics['sortino_ratio']) if all_metrics['sortino_ratio'] else 0,
        'avg_win_rate': sum(all_metrics['win_rate']) / len(all_metrics['win_rate']) if all_metrics['win_rate'] else 0,
        'avg_profit_factor': sum(all_metrics['profit_factor']) / len(all_metrics['profit_factor']) if all_metrics['profit_factor'] else 0,
        'avg_max_drawdown': sum(all_metrics['max_drawdown']) / len(all_metrics['max_drawdown']) if all_metrics['max_drawdown'] else 0,
        'avg_total_pnl': sum(all_metrics['total_pnl']) / len(all_metrics['total_pnl']) if all_metrics['total_pnl'] else 0,
        'total_trades': sum(all_metrics['total_trades']),
        'best_sharpe_symbol': max(results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))[0] if results else '',
        'worst_sharpe_symbol': min(results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))[0] if results else '',
        'best_pnl_symbol': max(results.items(), key=lambda x: x[1].get('total_pnl', 0))[0] if results else '',
        'worst_pnl_symbol': min(results.items(), key=lambda x: x[1].get('total_pnl', 0))[0] if results else '',
    }
    
    return summary


def save_results(backtest_results: dict, optimal_params: dict, recent_signals: dict, summary: dict):
    """
    Save backtest results, optimal parameters, and recent signals to files
    
    Args:
        backtest_results: Dictionary of backtest results
        optimal_params: Dictionary of optimal parameters
        recent_signals: Dictionary of recent signals
        summary: Dictionary of summary statistics
    """
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save backtest results
    with open(output_dir / f'nyse_backtest_results_{timestamp}.json', 'w') as f:
        json.dump(backtest_results, f, indent=2)
        
    # Save optimal parameters
    with open(output_dir / f'nyse_optimal_params_{timestamp}.json', 'w') as f:
        json.dump(optimal_params, f, indent=2)
        
    # Save recent signals
    with open(output_dir / f'nyse_recent_signals_{timestamp}.json', 'w') as f:
        # Convert datetime objects to strings
        signals_json = {}
        for symbol, signals in recent_signals.items():
            signals_json[symbol] = []
            for signal in signals:
                signal_copy = signal.copy()
                if 'Date' in signal_copy and isinstance(signal_copy['Date'], datetime):
                    signal_copy['Date'] = signal_copy['Date'].strftime('%Y-%m-%d %H:%M:%S')
                signals_json[symbol].append(signal_copy)
        json.dump(signals_json, f, indent=2)
        
    # Save summary
    with open(output_dir / f'nyse_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Results saved to output directory with timestamp {timestamp}")


def main():
    """Main execution function"""
    try:
        # 1. Load NYSE stock data
        stock_data = load_nyse_data(start_date='2020-01-01')
        if not stock_data:
            logger.error("No stock data loaded. Exiting.")
            return
            
        # 2. Optimize strategy parameters using a representative stock
        # Using the first stock as training data
        training_symbol = list(stock_data.keys())[0]
        logger.info(f"Using {training_symbol} as training data for optimization")
        training_data = stock_data[training_symbol]
        
        # Optimize strategy parameters
        optimal_params = optimize_strategy(training_data)
        if not optimal_params:
            logger.error("Strategy optimization failed. Exiting.")
            return
            
        # 3. Run parallel backtests with optimized parameters
        backtest_results = run_parallel_backtest(stock_data, optimal_params)
        
        # 4. Analyze recent trading signals (last 7 days)
        recent_signals = analyze_recent_signals(stock_data, optimal_params)
        
        # 5. Calculate performance summary
        summary = calculate_performance_summary(backtest_results)
        
        # 6. Save results
        save_results(backtest_results, optimal_params, recent_signals, summary)
        
        # 7. Display results summary
        logger.info("\n=== Backtest Results Summary ===")
        print(f"\nAverage Performance Metrics:")
        print(f"Sharpe Ratio: {summary['avg_sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {summary['avg_sortino_ratio']:.2f}")
        print(f"Win Rate: {summary['avg_win_rate']:.2%}")
        print(f"Profit Factor: {summary['avg_profit_factor']:.2f}")
        print(f"Max Drawdown: ${summary['avg_max_drawdown']:.2f}")
        print(f"Average P&L: ${summary['avg_total_pnl']:.2f}")
        print(f"Total Trades Executed: {summary['total_trades']}")
        
        print("\nBest Performing Stocks:")
        print(f"Best Sharpe Ratio: {summary['best_sharpe_symbol']}")
        print(f"Best Total P&L: {summary['best_pnl_symbol']}")
        
        print("\nRecent Trading Signals (Last 7 Days):")
        for symbol, signals in recent_signals.items():
            print(f"\n{symbol}:")
            for signal in signals:
                signal_type = signal.get('Type', 'UNKNOWN')
                price = signal.get('Price', 0.0)
                date = signal.get('Date', datetime.now()).strftime('%Y-%m-%d')
                score = signal.get('Score', 0.0)
                print(f"  {date}: {signal_type} at ${price:.2f} (Score: {score:.2f})")
        
    except Exception as e:
        logger.exception("An error occurred during execution")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimization.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Import project modules
from src.utils.exchange_data_loader import ExchangeDataLoader
from src.strategies.advanced_multi_strategy import AdvancedMultiStrategy
from src.strategies.backtester import Backtester


class ParallelStrategyOptimizer:
    """
    Optimizer for finding the best strategy parameters using parallel processing
    """
    
    def __init__(
        self, 
        stock_data: Dict[str, pd.DataFrame],
        train_size: float = 0.7,
        validation_size: float = 0.15,
        test_size: float = 0.15,
        optimization_metric: str = 'sharpe_ratio',
        initial_capital: float = 100000.0,
        n_workers: int = None
    ):
        """
        Initialize the optimizer
        
        Args:
            stock_data: Dictionary of stock data (symbol -> DataFrame)
            train_size: Proportion of data for training
            validation_size: Proportion of data for validation
            test_size: Proportion of data for testing
            optimization_metric: Metric to optimize ('sharpe_ratio', 'sortino_ratio', etc.)
            initial_capital: Initial capital for backtests
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.stock_data = stock_data
        self.train_size = train_size
        self.validation_size = validation_size 
        self.test_size = test_size
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        
        # Verify splits add up to 1.0
        total_split = train_size + validation_size + test_size
        if abs(total_split - 1.0) > 1e-6:
            logger.warning(f"Data splits sum to {total_split}, normalizing to 1.0")
            factor = 1.0 / total_split
            self.train_size *= factor
            self.validation_size *= factor
            self.test_size *= factor
        
        # Split the data for each stock
        self.train_data = {}
        self.validation_data = {}
        self.test_data = {}
        
        for symbol, data in self.stock_data.items():
            # Calculate split points
            n_samples = len(data)
            train_end = int(n_samples * self.train_size)
            valid_end = int(n_samples * (self.train_size + self.validation_size))
            
            # Split the data
            self.train_data[symbol] = data.iloc[:train_end]
            self.validation_data[symbol] = data.iloc[train_end:valid_end]
            self.test_data[symbol] = data.iloc[valid_end:]
            
            logger.debug(f"Split {symbol}: train={len(self.train_data[symbol])}, "
                         f"validation={len(self.validation_data[symbol])}, "
                         f"test={len(self.test_data[symbol])}")

        logger.info(f"Initialized optimizer with {len(stock_data)} stocks "
                   f"and {self.n_workers} workers")
        logger.info(f"Optimization metric: {optimization_metric}")
    
    def optimize(self, param_grid: Dict[str, List]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            param_grid: Dictionary of parameter ranges to search
            
        Returns:
            Tuple of (best_params, results_dict) where results_dict contains
            performance metrics for each parameter combination
        """
        logger.info("Starting optimization process...")
        
        # Generate all parameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Create all parameter dictionaries
        all_params = []
        for combo in param_combinations:
            param_dict = {key: value for key, value in zip(param_keys, combo)}
            all_params.append(param_dict)
        
        # Use a sample of stocks for optimization to speed up the process
        # Pick a random sample of stocks for validation
        sample_size = min(5, len(self.stock_data))
        selected_symbols = np.random.choice(list(self.stock_data.keys()), 
                                           size=sample_size, 
                                           replace=False)
        
        logger.info(f"Using {sample_size} stocks for optimization: {selected_symbols}")
        
        train_sample = {sym: self.train_data[sym] for sym in selected_symbols}
        validation_sample = {sym: self.validation_data[sym] for sym in selected_symbols}

        # Run backtests for all parameter combinations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create futures for training data
            futures = [
                executor.submit(self._evaluate_params, params, train_sample, 'train')
                for params in all_params
            ]
            
            # Collect training results
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in parameter evaluation: {str(e)}")
        
        # Find the best parameters based on training data
        if not results:
            logger.error("No valid results from optimization")
            return {}, {}
            
        # Sort results by metric value
        results.sort(key=lambda x: x[self.optimization_metric], reverse=True)
        
        # Get the top 5 parameter sets
        top_params = [r['params'] for r in results[:5]]
        logger.info(f"Top 5 parameter sets from training: {top_params}")
        
        # Evaluate the top parameter sets on validation data
        validation_results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create futures for validation data
            futures = [
                executor.submit(self._evaluate_params, params, validation_sample, 'validation')
                for params in top_params
            ]
            
            # Collect validation results
            for future in futures:
                try:
                    validation_results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in validation: {str(e)}")
        
        # Find the best parameters based on validation data
        if not validation_results:
            logger.error("No valid validation results")
            # Fall back to best training parameters
            best_params = results[0]['params'] if results else {}
        else:
            # Sort validation results by metric value
            validation_results.sort(key=lambda x: x[self.optimization_metric], reverse=True)
            best_params = validation_results[0]['params']
        
        logger.info(f"Best parameters: {best_params}")
        
        # Create a dictionary of all results
        results_dict = {
            'training_results': results,
            'validation_results': validation_results,
            'best_params': best_params
        }
        
        return best_params, results_dict
    
    def _evaluate_params(
        self, 
        params: Dict[str, Any], 
        stock_data: Dict[str, pd.DataFrame],
        phase: str
    ) -> Dict[str, Any]:
        """
        Evaluate a parameter set on the given data
        
        Args:
            params: Strategy parameters to test
            stock_data: Dictionary of stock data (symbol -> DataFrame)
            phase: 'train', 'validation', or 'test'
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            strategy = AdvancedMultiStrategy(**params)
            backtester = Backtester(strategy, initial_capital=self.initial_capital)
            
            # Run backtest on each stock
            all_metrics = []
            for symbol, data in stock_data.items():
                result = backtester.run(data)
                if result and hasattr(result, 'metrics'):
                    all_metrics.append(result.metrics)
            
            # Calculate average metrics across all stocks
            if not all_metrics:
                logger.warning(f"No valid metrics for parameter set: {params}")
                return {
                    'params': params,
                    'phase': phase,
                    self.optimization_metric: float('-inf'),
                    'avg_metrics': {},
                    'n_stocks': 0
                }
            
            # Calculate average metrics
            avg_metrics = self._calculate_average_metrics(all_metrics)
            
            return {
                'params': params,
                'phase': phase,
                self.optimization_metric: avg_metrics.get(self.optimization_metric, float('-inf')),
                'avg_metrics': avg_metrics,
                'n_stocks': len(all_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return {
                'params': params,
                'phase': phase,
                self.optimization_metric: float('-inf'),
                'avg_metrics': {},
                'n_stocks': 0
            }
    
    def _calculate_average_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average metrics across multiple backtest results
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with average metrics
        """
        if not metrics_list:
            return {}
            
        # Initialize with the keys from the first metrics dict
        avg_metrics = {key: [] for key in metrics_list[0] if isinstance(metrics_list[0][key], (int, float))}
        
        # Collect values for each metric
        for metrics in metrics_list:
            for key in avg_metrics:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    avg_metrics[key].append(metrics[key])
        
        # Calculate averages
        result = {}
        for key, values in avg_metrics.items():
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
                
        return result
    
    def evaluate_best_params(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the best parameters on the test data
        
        Args:
            best_params: Best parameters from optimization
            
        Returns:
            Dictionary with test performance metrics
        """
        logger.info("Evaluating best parameters on test data...")
        test_result = self._evaluate_params(best_params, self.test_data, 'test')
        
        logger.info(f"Test performance with best parameters:")
        for key, value in test_result['avg_metrics'].items():
            logger.info(f"  {key}: {value}")
            
        return test_result


def load_nyse_data(start_date: str = '2020-01-01', max_symbols: int = 30) -> dict:
    """
    Load NYSE stock data from the data/NYSE folder
    
    Args:
        start_date: Start date for the data in format 'YYYY-MM-DD'
        max_symbols: Maximum number of symbols to load (for efficiency)
        
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
    
    # Limit number of symbols for optimization
    if max_symbols and len(symbol_files) > max_symbols:
        symbol_files = symbol_files[:max_symbols]
    
    logger.info(f"Loading data for {len(symbol_files)} NYSE symbols")
    
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


def create_param_grid() -> Dict[str, List]:
    """
    Create parameter grid for optimization
    
    Returns:
        Dictionary with parameter ranges
    """
    return {
        # Moving Average parameters
        'sma_short_period': [10, 20, 30],
        'sma_long_period': [50, 100],
        'ema_short_period': [8, 12, 16],
        'ema_long_period': [21, 26, 30],
        
        # RSI parameters
        'rsi_period': [7, 14, 21],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        
        # Bollinger Bands parameters
        'bb_period': [15, 20, 25],
        'bb_std': [1.8, 2.0, 2.2],
        
        # MACD parameters
        'macd_fast': [8, 12, 16],
        'macd_slow': [21, 26, 30],
        'macd_signal': [7, 9, 11],
        
        # ADX parameters
        'adx_period': [10, 14, 18],
        'adx_threshold': [20, 25, 30],
        
        # Volume parameters
        'volume_ma_period': [10, 20, 30],
        'volume_threshold': [1.2, 1.5, 1.8],
        
        # ATR parameters
        'atr_period': [10, 14, 18],
        'atr_multiplier': [1.5, 2.0, 2.5],
        'trailing_stop_factor': [2.0, 3.0, 4.0],
        
        # Strategy weights must add up to 1.0, so we use a fixed set
        'trend_weight': [0.25],
        'momentum_weight': [0.25],
        'volatility_weight': [0.25],
        'volume_weight': [0.25],
        
        # Signal thresholds
        'buy_threshold': [0.4, 0.5, 0.6],
        'sell_threshold': [-0.4, -0.5, -0.6],
        
        # Strategy parameters
        'min_score_buy': [2, 3, 4],
        'min_score_sell': [2, 3, 4],
        'risk_percent': [1.0],
        'profit_target_factor': [1.5, 2.0, 2.5]
    }


def plot_optimization_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Plot optimization results
    
    Args:
        results: Dictionary of optimization results
        output_dir: Directory for output plots
    """
    output_dir.mkdir(exist_ok=True)
    
    # Extract training results
    training_results = results['training_results']
    
    if not training_results:
        logger.warning("No training results to plot")
        return
    
    # Sort by optimization metric
    training_results.sort(key=lambda x: x['avg_metrics'].get('sharpe_ratio', 0), reverse=True)
    
    # Extract metrics for plotting
    sharpe_values = [r['avg_metrics'].get('sharpe_ratio', 0) for r in training_results]
    sortino_values = [r['avg_metrics'].get('sortino_ratio', 0) for r in training_results]
    win_rates = [r['avg_metrics'].get('win_rate', 0) * 100 for r in training_results]  # Convert to percentage
    profit_factors = [r['avg_metrics'].get('profit_factor', 0) for r in training_results]
    
    # Plot Sharpe vs Sortino ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(sharpe_values, sortino_values, alpha=0.7)
    plt.title('Sharpe Ratio vs Sortino Ratio for Different Parameters')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Sortino Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'sharpe_vs_sortino.png')
    plt.close()
    
    # Plot Win Rate vs Profit Factor
    plt.figure(figsize=(10, 6))
    plt.scatter(win_rates, profit_factors, alpha=0.7)
    plt.title('Win Rate vs Profit Factor for Different Parameters')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Profit Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'winrate_vs_profitfactor.png')
    plt.close()
    
    # Plot distribution of Sharpe ratios
    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_values, bins=20, alpha=0.7)
    plt.title('Distribution of Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'sharpe_distribution.png')
    plt.close()
    
    # Plot top parameter combinations and their metrics
    top_n = min(10, len(training_results))
    top_results = training_results[:top_n]
    
    metrics = ['sharpe_ratio', 'sortino_ratio', 'win_rate', 'profit_factor', 'max_drawdown']
    
    # Create a table for the top parameters
    param_table = []
    for i, result in enumerate(top_results):
        row = [f"Combo {i+1}"]
        for metric in metrics:
            if metric == 'win_rate':
                # Convert win rate to percentage
                value = result['avg_metrics'].get(metric, 0) * 100
                row.append(f"{value:.1f}%")
            elif metric == 'max_drawdown':
                value = result['avg_metrics'].get(metric, 0)
                row.append(f"${value:.2f}")
            else:
                value = result['avg_metrics'].get(metric, 0)
                row.append(f"{value:.2f}")
        param_table.append(row)
    
    # Plot the table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(
        cellText=param_table,
        colLabels=["Parameter Set"] + metrics,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Top Performing Parameter Combinations')
    plt.savefig(output_dir / 'top_parameters.png')
    plt.close()


def save_results(
    best_params: Dict[str, Any], 
    results: Dict[str, Any],
    test_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Save optimization results to files
    
    Args:
        best_params: Best parameters from optimization
        results: Dictionary of optimization results
        test_results: Test performance results
        output_dir: Directory for output files
    """
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best parameters
    with open(output_dir / f'optimal_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Extract a simplified version of results for saving
    simplified_results = {
        'training_metrics': [
            {
                'params': r['params'],
                'metrics': r['avg_metrics']
            }
            for r in results['training_results'][:20]  # Save only top 20
        ],
        'validation_metrics': [
            {
                'params': r['params'],
                'metrics': r['avg_metrics']
            }
            for r in results['validation_results']
        ],
        'test_metrics': test_results['avg_metrics'] if 'avg_metrics' in test_results else {}
    }
    
    # Save results
    with open(output_dir / f'optimization_results_{timestamp}.json', 'w') as f:
        json.dump(simplified_results, f, indent=2)


def main():
    """Main function for strategy optimization"""
    try:
        logger.info("Starting strategy optimization process")
        
        # 1. Load NYSE stock data
        stock_data = load_nyse_data(start_date='2020-01-01')
        if not stock_data:
            logger.error("No stock data loaded. Exiting.")
            return
        
        # 2. Create parameter grid
        param_grid = create_param_grid()
        logger.info(f"Created parameter grid with {len(param_grid)} parameters")
        
        # 3. Initialize optimizer
        optimizer = ParallelStrategyOptimizer(
            stock_data=stock_data,
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            optimization_metric='sharpe_ratio',
            initial_capital=100000.0
        )
        
        # 4. Run optimization
        best_params, results = optimizer.optimize(param_grid)
        
        # 5. Evaluate on test data
        test_results = optimizer.evaluate_best_params(best_params)
        
        # 6. Create output directory
        output_dir = Path('output/optimization')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 7. Save results
        save_results(best_params, results, test_results, output_dir)
        
        # 8. Plot results
        plot_optimization_results(results, output_dir)
        
        logger.info("Optimization completed successfully")
        
        # 9. Print summary
        print("\n=== Optimization Results ===")
        print(f"\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print("\nTest Performance:")
        for metric, value in test_results['avg_metrics'].items():
            if metric == 'win_rate':
                print(f"  {metric}: {value:.2%}")
            elif metric in ['sharpe_ratio', 'sortino_ratio', 'profit_factor']:
                print(f"  {metric}: {value:.2f}")
            elif metric == 'max_drawdown':
                print(f"  {metric}: ${value:.2f}")
            else:
                print(f"  {metric}: {value}")
        
    except Exception as e:
        logger.exception("An error occurred during optimization")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Parameter Optimization Test Script

This script performs the following steps:
1. Runs a backtest for a set of symbols
2. Analyzes the results to identify optimal parameters for each symbol
3. Saves the optimized parameters to the parameter storage database
4. Runs a second backtest with the optimized parameters to validate improvements
"""

import json
import sys
import os
import logging
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from itertools import product

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_system.market_backtest import run_market_backtest
from src.market_system.storage import ParameterStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config/market/mid-mrk-momentum-master.json"
START_DATE = "2024-01-01"
END_DATE = "2024-03-10"
TIMEFRAME = "1d"
EXECUTION_MODE = "parallel"
WORKERS = 4
PARAM_DB_PATH = "data/strategy_params.db"
REPORT_DIR = "reports/backtest/parameter_optimization"

def load_symbols(config_file, max_symbols=10):
    """Load symbols from configuration file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            symbols = config.get('symbols', [])[:max_symbols]
            
        if not symbols:
            logger.error(f"No symbols found in configuration file: {config_file}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(symbols)} symbols for parameter optimization")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {config_file}: {str(e)}")
        sys.exit(1)

def create_parameter_grid():
    """Create a grid of parameters to test"""
    param_grid = {
        'fast_period': [5, 8, 10, 12, 15],
        'slow_period': [20, 25, 30, 35, 40],
        'signal_threshold': [0.0, 0.001, 0.002],
        'volume_threshold': [1.0, 1.5, 2.0],
        'rsi_period': [9, 14, 21],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35]
    }
    
    return param_grid

def run_optimization_backtest(symbols, param_sets, iteration):
    """Run backtest with a specific parameter set"""
    strategies = ["moving_average_crossover"]
    
    # Directory for this optimization run
    report_subdir = f"{REPORT_DIR}/iteration_{iteration}"
    os.makedirs(report_subdir, exist_ok=True)
    
    results = []
    
    for params in param_sets:
        # Create a unique name for this parameter set
        param_name = f"MAC_F{params['fast_period']}_S{params['slow_period']}_V{params['volume_threshold']}_RSI{params['rsi_period']}"
        logger.info(f"Running backtest for parameter set: {param_name}")
        
        # Run backtest with specific parameters
        run_result = run_market_backtest(
            symbols=symbols,
            strategies=strategies,
            start_date=START_DATE,
            end_date=END_DATE,
            timeframe=TIMEFRAME,
            execution_mode=EXECUTION_MODE,
            workers=WORKERS,
            generate_reports=True,
            reports_dir=f"{report_subdir}/{param_name}",
            custom_params={
                'fast_period': params['fast_period'],
                'slow_period': params['slow_period'],
                'signal_threshold': params['signal_threshold'],
                'volume_threshold': params['volume_threshold'],
                'rsi_period': params['rsi_period'],
                'rsi_overbought': params['rsi_overbought'],
                'rsi_oversold': params['rsi_oversold'],
                'use_symbol_specific_params': False,  # Disable for optimization run
                'param_db_path': PARAM_DB_PATH
            }
        )
        
        # Process and store results
        for symbol, metrics in run_result.items():
            results.append({
                'symbol': symbol,
                'param_set': param_name,
                'parameters': params,
                'final_equity': metrics.get('final_equity', 0),
                'return': metrics.get('return', 0),
                'sharpe': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0)
            })
    
    return results

def find_best_parameters(results):
    """Find the best parameters for each symbol based on a weighted scoring function"""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.warning("No results found to analyze")
        return {}
    
    # Group by symbol
    symbol_groups = df.groupby('symbol')
    
    best_params = {}
    
    for symbol, group in symbol_groups:
        # Skip symbols with no trades
        if group['total_trades'].max() == 0:
            logger.info(f"Skipping {symbol} - no trades executed")
            continue
        
        # Create a scoring function that considers multiple metrics
        # Higher is better for: return, sharpe, win_rate, total_trades
        # Lower is better for: max_drawdown
        
        # Normalize values to 0-1 range for each metric
        normalized = group.copy()
        
        # Find positive returns first
        positive_returns = group[group['return'] > 0]
        
        if not positive_returns.empty:
            # Prioritize positive returns first
            best_row = positive_returns.loc[positive_returns['return'].idxmax()]
        else:
            # If no positive returns, normalize and score
            for col in ['return', 'sharpe', 'max_drawdown', 'win_rate', 'total_trades']:
                if col in normalized.columns:
                    min_val = normalized[col].min()
                    max_val = normalized[col].max()
                    range_val = max_val - min_val
                    
                    if range_val > 0:
                        if col == 'max_drawdown':
                            # Lower is better for drawdown
                            normalized[col] = 1 - ((normalized[col] - min_val) / range_val)
                        else:
                            # Higher is better for the rest
                            normalized[col] = (normalized[col] - min_val) / range_val
                    else:
                        normalized[col] = 0.5  # Set to neutral if no variance
            
            # Calculate weighted score - adjust weights as needed
            normalized['score'] = (
                normalized['return'] * 0.4 +
                normalized['sharpe'] * 0.2 +
                normalized['max_drawdown'] * 0.15 +
                normalized['win_rate'] * 0.15 +
                normalized['total_trades'] * 0.1
            )
            
            # Get row with highest score
            best_row = group.loc[normalized['score'].idxmax()]
        
        # Store best parameters
        best_params[symbol] = {
            'parameters': best_row['parameters'],
            'metrics': {
                'return': best_row['return'],
                'sharpe': best_row['sharpe'],
                'max_drawdown': best_row['max_drawdown'],
                'win_rate': best_row['win_rate'],
                'total_trades': best_row['total_trades']
            }
        }
        
        logger.info(f"Best parameters for {symbol}: {best_row['parameters']}")
        logger.info(f"Metrics: Return: {best_row['return']:.2f}%, Win Rate: {best_row['win_rate']:.2f}%, Trades: {best_row['total_trades']}")
    
    return best_params

def save_optimized_parameters(best_params):
    """Save optimized parameters to storage"""
    try:
        param_storage = ParameterStorage(PARAM_DB_PATH)
        
        for symbol, data in best_params.items():
            strategy_name = "MAC_optimizer"
            parameters = data['parameters']
            metrics = data['metrics']
            
            # Save parameters
            param_storage.set_parameters(strategy_name, symbol, parameters)
            
            # Record performance
            param_storage.record_performance(strategy_name, symbol, parameters, metrics)
            
            logger.info(f"Saved optimized parameters for {symbol}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving optimized parameters: {str(e)}")
        return False

def run_validation_backtest(symbols):
    """Run a backtest with optimized parameters"""
    strategies = ["moving_average_crossover"]
    
    # Run backtest with optimized parameters
    logger.info("Running validation backtest with optimized parameters")
    
    validation_result = run_market_backtest(
        symbols=symbols,
        strategies=strategies,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        execution_mode=EXECUTION_MODE,
        workers=WORKERS,
        generate_reports=True,
        reports_dir=f"{REPORT_DIR}/validation",
        custom_params={
            'use_symbol_specific_params': True,  # Enable using stored parameters
            'param_db_path': PARAM_DB_PATH
        }
    )
    
    # Analyze and report results
    total_return = sum(metric.get('return', 0) for metric in validation_result.values()) / len(validation_result)
    positive_returns = sum(1 for metric in validation_result.values() if metric.get('return', 0) > 0)
    total_symbols = len(validation_result)
    
    logger.info(f"Validation backtest complete")
    logger.info(f"Average return: {total_return:.2f}%")
    logger.info(f"Symbols with positive returns: {positive_returns}/{total_symbols} ({positive_returns/total_symbols*100:.2f}%)")
    
    return validation_result

def main():
    """Main function to run the parameter optimization process"""
    # Create report directory
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Step 1: Load symbols
    symbols = load_symbols(CONFIG_FILE)
    logger.info(f"Starting parameter optimization for {len(symbols)} symbols")
    
    # Step 2: Create parameter grid
    param_grid = create_parameter_grid()
    
    # Create a subset of parameter combinations to test (full grid may be too large)
    # This is a simplified approach - more sophisticated methods like Bayesian optimization could be used
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total parameter combinations: {total_combinations}")
    
    # If too many combinations, select a subset
    max_combinations = 10  # Adjust based on time constraints
    if total_combinations > max_combinations:
        # Select a subset of key parameters and combinations
        fast_periods = param_grid['fast_period']
        slow_periods = param_grid['slow_period'][::2]  # Take every second value
        volume_thresholds = param_grid['volume_threshold'][::2]  # Take every second value
        
        param_sets = []
        for fast, slow, vol in product(fast_periods, slow_periods, volume_thresholds):
            if slow > fast * 1.5:  # Ensure meaningful separation between fast and slow
                param_sets.append({
                    'fast_period': fast,
                    'slow_period': slow,
                    'signal_threshold': param_grid['signal_threshold'][1],  # Middle value
                    'volume_threshold': vol,
                    'rsi_period': param_grid['rsi_period'][1],  # Middle value
                    'rsi_overbought': param_grid['rsi_overbought'][1],  # Middle value
                    'rsi_oversold': param_grid['rsi_oversold'][1]  # Middle value
                })
        
        logger.info(f"Testing {len(param_sets)} parameter combinations")
    else:
        # Use all combinations
        param_sets = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
        logger.info(f"Testing all {len(param_sets)} parameter combinations")
    
    # Step 3: Run optimization backtest
    results = run_optimization_backtest(symbols, param_sets, iteration=1)
    
    # Step 4: Find the best parameters for each symbol
    best_params = find_best_parameters(results)
    
    # Step 5: Save the optimized parameters
    if save_optimized_parameters(best_params):
        logger.info("Successfully saved optimized parameters")
    else:
        logger.error("Failed to save optimized parameters")
        return
    
    # Step 6: Run validation backtest with optimized parameters
    validation_result = run_validation_backtest(symbols)
    
    # Step 7: Report summary
    logger.info("Parameter optimization process completed")
    logger.info(f"Check reports at: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    logger.info(f"Total runtime: {end_time - start_time}") 
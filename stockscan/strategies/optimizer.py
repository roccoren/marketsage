"""
Strategy optimizer for backtesting

This module provides functionality to optimize strategy parameters using various methods
including grid search, random search, and Bayesian optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Callable, Tuple, Optional
import itertools
import logging
import time
from sklearn.model_selection import train_test_split
import random

from src.strategies.backtester import Backtester

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizer for trading strategies
    
    This class performs parameter optimization for trading strategies using
    different optimization methods. It splits data into training and testing sets,
    evaluates strategies across parameter combinations, and returns the best parameters.
    """
    
    def __init__(self, strategy_class: Any, param_grid: Dict[str, List[Any]], 
                 data: pd.DataFrame, train_size: float = 0.7, 
                 metric: str = 'sharpe_ratio', random_state: int = 42):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        strategy_class : Any
            The strategy class to optimize
        param_grid : Dict[str, List[Any]]
            Dictionary of parameter names and possible values
        data : pd.DataFrame
            Historical price data for optimization
        train_size : float, optional
            Proportion of data to use for training, by default 0.7
        metric : str, optional
            Metric to optimize for, by default 'sharpe_ratio'
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_size = train_size
        self.metric = metric
        self.random_state = random_state
        self.results = pd.DataFrame()
        
        # Split data into train/test sets
        train_idx = int(len(data) * train_size)
        self.train_data = data.iloc[:train_idx].copy()
        self.test_data = data.iloc[train_idx:].copy()
        
        logger.info(f"Train data: {len(self.train_data)} records from {self.train_data.index[0]} to {self.train_data.index[-1]}")
        logger.info(f"Test data: {len(self.test_data)} records from {self.test_data.index[0]} to {self.test_data.index[-1]}")
        
        # Define optimization objective based on metric
        self.metric_functions = {
            'sharpe_ratio': lambda result: result.metrics.get('sharpe_ratio', -999),
            'sortino_ratio': lambda result: result.metrics.get('sortino_ratio', -999),
            'profit_factor': lambda result: result.metrics.get('profit_factor', 0),
            'net_profit': lambda result: result.metrics.get('net_profit', 0),
            'net_profit_pct': lambda result: result.metrics.get('net_profit_pct', 0),
            'max_drawdown': lambda result: -result.metrics.get('max_drawdown', 100),  # Negative because we want to maximize
            'win_rate': lambda result: result.metrics.get('win_rate', 0),
            'expectancy': lambda result: result.metrics.get('expectancy', -999),
            'calmar_ratio': lambda result: result.metrics.get('calmar_ratio', -999),
            'custom_score': self._custom_score
        }
        
        if metric not in self.metric_functions:
            logger.warning(f"Unknown metric '{metric}', defaulting to 'sharpe_ratio'")
            self.metric = 'sharpe_ratio'
        
        self.objective_function = self.metric_functions[self.metric]
    
    def _custom_score(self, result: Any) -> float:
        """
        Calculate a custom scoring function combining multiple metrics
        
        This is an example of a custom score that balances returns, risk, and consistency
        
        Parameters:
        -----------
        result : Any
            Backtest result object
            
        Returns:
        --------
        float
            Custom score value
        """
        if not result or not hasattr(result, 'metrics'):
            return -999
        
        metrics = result.metrics
        
        # Balance of returns and risk
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        profit_factor = metrics.get('profit_factor', 0)
        win_rate = metrics.get('win_rate', 0)
        drawdown = metrics.get('max_drawdown', 100)
        
        # Penalize strategies with too few trades
        num_trades = metrics.get('total_trades', 0)
        trade_factor = min(1.0, num_trades / 10)  # Scale up to 10 trades
        
        # Prefer strategies with higher win rates
        win_factor = win_rate * 2  # Give more weight to win rate
        
        # Drawdown penalty (transformed to be positive)
        dd_factor = max(0, 1 - (drawdown / 100))
        
        # Combined score
        score = (sharpe * 0.4) + (sortino * 0.2) + (profit_factor * 0.1) + (win_factor * 0.2) + (dd_factor * 0.1)
        
        # Apply trade count penalty
        score = score * trade_factor
        
        return score
    
    def evaluate_params(self, params: Dict[str, Any], data: pd.DataFrame, 
                       symbol: str = 'OPTIMIZATION', initial_capital: float = 100000) -> float:
        """
        Evaluate a set of parameters on given data
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Parameters to evaluate
        data : pd.DataFrame
            Data to run backtest on
        symbol : str, optional
            Symbol name for logging, by default 'OPTIMIZATION'
        initial_capital : float, optional
            Initial capital for backtest, by default 100000
            
        Returns:
        --------
        float
            Score based on objective function
        """
        try:
            # Instantiate strategy with parameters
            strategy = self.strategy_class(params=params)
            
            # Create backtester
            backtester = Backtester(
                strategy=strategy,
                initial_capital=initial_capital,
                position_size=1.0  # Use full position size for optimization
            )
            
            # Run backtest
            result = backtester.run(
                data=data,
                symbol=symbol
            )
            
            if result and hasattr(result, 'metrics'):
                # Calculate objective
                score = self.objective_function(result)
                
                # Additional metrics to track
                metrics = {
                    'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': result.metrics.get('sortino_ratio', 0),
                    'net_profit_pct': result.metrics.get('net_profit_pct', 0),
                    'max_drawdown': result.metrics.get('max_drawdown', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'total_trades': result.metrics.get('total_trades', 0),
                    'profit_factor': result.metrics.get('profit_factor', 0)
                }
                
                return score, metrics
            else:
                logger.warning(f"Backtest failed for parameters: {params}")
                return -999, {}
                
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return -999, {}
    
    def grid_search(self, params_to_try: List[Dict[str, Any]], max_iterations: int = None, 
                   initial_capital: float = 100000) -> pd.DataFrame:
        """
        Perform grid search over parameter combinations
        
        Parameters:
        -----------
        params_to_try : List[Dict[str, Any]]
            List of parameter dictionaries to evaluate
        max_iterations : int, optional
            Maximum number of iterations, by default None (try all)
        initial_capital : float, optional
            Initial capital for backtest, by default 100000
            
        Returns:
        --------
        pd.DataFrame
            Results of parameter optimization
        """
        results = []
        total = len(params_to_try)
        
        if max_iterations and max_iterations < total:
            logger.info(f"Limiting grid search to {max_iterations} iterations out of {total} possible combinations")
            # Randomly select parameter combinations if limited
            random.seed(self.random_state)
            params_to_try = random.sample(params_to_try, max_iterations)
            total = max_iterations
        
        logger.info(f"Starting grid search with {total} parameter combinations")
        start_time = time.time()
        
        for i, params in enumerate(params_to_try):
            logger.info(f"Evaluating parameters {i+1}/{total}: {params}")
            
            score, metrics = self.evaluate_params(params, self.train_data, initial_capital=initial_capital)
            
            # Store result
            result = {
                'iteration': i + 1,
                'params': params,
                'train_score': score,
                **{f'train_{k}': v for k, v in metrics.items()}
            }
            
            # Evaluate best parameters on test data
            if score > -999:  # Only evaluate valid results on test data
                test_score, test_metrics = self.evaluate_params(params, self.test_data, initial_capital=initial_capital)
                result['test_score'] = test_score
                for k, v in test_metrics.items():
                    result[f'test_{k}'] = v
            
            results.append(result)
            
            # Progress logging
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            logger.info(f"Completed {i+1}/{total} iterations ({(i+1)/total*100:.1f}%). "
                      f"Elapsed: {elapsed/60:.1f} min, Estimated remaining: {remaining/60:.1f} min")
        
        # Convert results to DataFrame
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values('train_score', ascending=False)
        
        return self.results
    
    def random_search(self, num_iterations: int = 10, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Perform random search over parameter combinations
        
        Parameters:
        -----------
        num_iterations : int, optional
            Number of random parameter combinations to try, by default 10
        initial_capital : float, optional
            Initial capital for backtest, by default 100000
            
        Returns:
        --------
        pd.DataFrame
            Results of parameter optimization
        """
        random.seed(self.random_state)
        
        # Generate random parameter combinations
        params_to_try = []
        for _ in range(num_iterations):
            params = {}
            for param_name, param_values in self.param_grid.items():
                if isinstance(param_values, list):
                    params[param_name] = random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Handle ranges (min, max)
                    min_val, max_val = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        params[param_name] = random.uniform(min_val, max_val)
            params_to_try.append(params)
        
        logger.info(f"Starting random search with {num_iterations} iterations")
        
        # Run grid search with generated parameters
        return self.grid_search(params_to_try, initial_capital=initial_capital)
    
    def bayesian_optimization(self, num_iterations: int = 10, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Perform Bayesian optimization for parameter tuning
        
        Note: This method requires the 'scikit-optimize' package
        
        Parameters:
        -----------
        num_iterations : int, optional
            Number of iterations, by default 10
        initial_capital : float, optional
            Initial capital for backtest, by default 100000
            
        Returns:
        --------
        pd.DataFrame
            Results of parameter optimization
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("Bayesian optimization requires scikit-optimize. Please install it with: pip install scikit-optimize")
            return pd.DataFrame()
        
        # Define search space
        dimensions = []
        dim_names = []
        
        for param_name, param_values in self.param_grid.items():
            if isinstance(param_values, list):
                if all(isinstance(x, int) for x in param_values):
                    dimensions.append(Integer(min(param_values), max(param_values), name=param_name))
                elif all(isinstance(x, float) for x in param_values):
                    dimensions.append(Real(min(param_values), max(param_values), name=param_name))
                else:
                    dimensions.append(Categorical(param_values, name=param_name))
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                min_val, max_val = param_values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    dimensions.append(Integer(min_val, max_val, name=param_name))
                else:
                    dimensions.append(Real(min_val, max_val, name=param_name))
            
            dim_names.append(param_name)
        
        # Define objective function for skopt
        @use_named_args(dimensions=dimensions)
        def objective_wrapper(**params):
            score, _ = self.evaluate_params(params, self.train_data, initial_capital=initial_capital)
            return -score  # Negative because skopt minimizes
        
        # Run Bayesian optimization
        logger.info(f"Starting Bayesian optimization with {num_iterations} iterations")
        start_time = time.time()
        
        result = gp_minimize(
            objective_wrapper,
            dimensions=dimensions,
            n_calls=num_iterations,
            random_state=self.random_state,
            verbose=True
        )
        
        # Process results
        results = []
        for i, (params_list, score) in enumerate(zip(result.x_iters, result.func_vals)):
            params = {dim.name: value for dim, value in zip(dimensions, params_list)}
            
            # Evaluate on train data
            train_score, train_metrics = self.evaluate_params(params, self.train_data, initial_capital=initial_capital)
            
            # Evaluate on test data
            test_score, test_metrics = self.evaluate_params(params, self.test_data, initial_capital=initial_capital)
            
            results.append({
                'iteration': i + 1,
                'params': params,
                'train_score': train_score,
                'test_score': test_score,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })
        
        # Add best result
        best_params = {dim.name: value for dim, value in zip(dimensions, result.x)}
        results.append({
            'iteration': num_iterations + 1,
            'params': best_params,
            'train_score': -result.fun,  # Negate back
            'is_best': True
        })
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values('train_score', ascending=False)
        
        return self.results
    
    def optimize(self, method: str = 'grid', iterations: int = 10, 
                initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Optimize strategy parameters using specified method
        
        Parameters:
        -----------
        method : str, optional
            Optimization method ('grid', 'random', or 'bayesian'), by default 'grid'
        iterations : int, optional
            Number of iterations for random and Bayesian methods, by default 10
        initial_capital : float, optional
            Initial capital for backtest, by default 100000
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results containing best parameters and scores
        """
        if method == 'grid':
            # Generate parameter combinations for grid search
            keys = list(self.param_grid.keys())
            values = list(self.param_grid.values())
            combinations = list(itertools.product(*values))
            params_list = [dict(zip(keys, combo)) for combo in combinations]
            
            logger.info(f"Grid search: {len(params_list)} parameter combinations generated")
            
            # Run with max iterations
            self.grid_search(params_list, max_iterations=iterations, initial_capital=initial_capital)
            
        elif method == 'random':
            self.random_search(num_iterations=iterations, initial_capital=initial_capital)
            
        elif method == 'bayesian':
            self.bayesian_optimization(num_iterations=iterations, initial_capital=initial_capital)
            
        else:
            logger.error(f"Unknown optimization method: {method}")
            return {}
        
        # Get best parameters
        if len(self.results) > 0:
            best_row = self.results.iloc[0]
            best_params = best_row['params']
            
            best_train_score = best_row['train_score']
            best_test_score = best_row['test_score'] if 'test_score' in best_row else None
            
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best train score ({self.metric}): {best_train_score:.4f}")
            if best_test_score is not None:
                logger.info(f"Best test score ({self.metric}): {best_test_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_train_score': best_train_score,
                'best_test_score': best_test_score
            }
        else:
            logger.warning("No valid results found during optimization")
            return {
                'best_params': {},
                'best_train_score': -999,
                'best_test_score': -999
            }
    
    def get_optimization_report(self) -> pd.DataFrame:
        """
        Get a DataFrame with optimization results
        
        Returns:
        --------
        pd.DataFrame
            Optimization results with flattenened parameters
        """
        if len(self.results) == 0:
            logger.warning("No optimization results available")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        report = self.results.copy()
        
        # Expand parameters
        param_columns = []
        for i, row in report.iterrows():
            params = row['params']
            for param_name, param_value in params.items():
                if param_name not in report.columns:
                    report[param_name] = None
                    param_columns.append(param_name)
                report.at[i, param_name] = param_value
        
        # Reorder columns
        cols = ['iteration', *param_columns, 'train_score', 'test_score']
        for col in report.columns:
            if col not in cols and not col == 'params':
                cols.append(col)
        
        # Select columns that exist
        existing_cols = [col for col in cols if col in report.columns]
        
        return report[existing_cols]
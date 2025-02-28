import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from itertools import product
from .backtester import Backtester
from .base import Strategy
import logging

# Configure logging
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    def __init__(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        train_size: float = 0.8,
        metric: str = 'sharpe_ratio'
    ):
        """
        Initialize the optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameters and their possible values
            data: Historical price data
            train_size: Fraction of data to use for training
            metric: Metric to optimize ('sharpe_ratio', 'profit_factor', etc.)
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_size = train_size
        self.metric = metric
        
        # Split data into train/test sets
        train_idx = int(len(data) * train_size)
        self.train_data = data.iloc[:train_idx]
        self.test_data = data.iloc[train_idx:]
        
        logger.info(f"Train data shape: {self.train_data.shape}")
        logger.info(f"Test data shape: {self.test_data.shape}")
        
        self.best_params = None
        self.best_train_score = float('-inf')
        self.best_test_score = None
        self.results = []

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations from param_grid"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(product(*param_values))
        logger.info(f"Generated {len(combinations)} parameter combinations to test")
        
        return [dict(zip(param_names, combo)) for combo in combinations]

    def _evaluate_strategy(self, strategy: Strategy, data: pd.DataFrame) -> float:
        """Evaluate strategy performance on given data"""
        try:
            backtester = Backtester(strategy, initial_capital=100000)
            result = backtester.run(data)
            score = result.metrics.get(self.metric, float('-inf'))
            
            logger.debug(f"Strategy metrics: {result.metrics}")
            
            if score == float('-inf'):
                logger.warning(f"Metric '{self.metric}' not found in results")
            
            return score
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            return float('-inf')

    def optimize(self, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run optimization process.
        
        Returns:
            Dictionary containing best parameters and their performance metrics
        """
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        logger.info(f"Starting optimization with {total_combinations} combinations...")
        
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"\nTesting combination {i}/{total_combinations}")
            logger.info(f"Parameters: {params}")
            
            try:
                # Create strategy instance with current parameters
                strategy = self.strategy_class(**params)
                
                # Run backtest on training data
                train_score = self._evaluate_strategy(strategy, self.train_data)
                logger.info(f"Training {self.metric}: {train_score:.4f}")
                
                # Run backtest on test data
                test_score = self._evaluate_strategy(strategy, self.test_data)
                logger.info(f"Test {self.metric}: {test_score:.4f}")
                
                self.results.append({
                    'params': params,
                    'train_score': train_score,
                    'test_score': test_score
                })
                
                # Update best parameters if we have better training score
                if train_score > self.best_train_score:
                    logger.info(f"New best score found: {train_score:.4f} (previous: {self.best_train_score:.4f})")
                    self.best_train_score = train_score
                    self.best_test_score = test_score
                    self.best_params = params
                
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {str(e)}", exc_info=True)
                continue
        
        if self.best_params is None:
            logger.warning("No valid parameter combinations found!")
        else:
            logger.info("\nOptimization complete!")
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best training {self.metric}: {self.best_train_score:.4f}")
            logger.info(f"Corresponding test {self.metric}: {self.best_test_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_train_score': self.best_train_score,
            'best_test_score': self.best_test_score,
            'all_results': self.results
        }

    def get_optimization_report(self) -> pd.DataFrame:
        """
        Generate a DataFrame with all optimization results.
        """
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Expand parameters into separate columns
        param_df = pd.DataFrame(list(df['params']))
        
        # Combine parameter columns with scores
        result_df = pd.concat([
            param_df,
            df['train_score'],
            df['test_score']
        ], axis=1)
        
        # Sort by training score descending
        result_df = result_df.sort_values('train_score', ascending=False)
        
        return result_df

    def plot_optimization_results(
        self,
        param_name: str,
        metric: str = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot parameter values vs performance metric.
        Useful for visualizing parameter sensitivity.
        
        Args:
            param_name: Parameter to plot on x-axis
            metric: Metric to plot on y-axis (defaults to optimization metric)
            figsize: Figure size tuple
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
            
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
            
        metric = metric or self.metric
        df = self.get_optimization_report()
        
        plt.figure(figsize=figsize)
        plt.scatter(df[param_name], df['train_score'], alpha=0.5, label='Train')
        plt.scatter(df[param_name], df['test_score'], alpha=0.5, label='Test')
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.title(f'{param_name} vs {metric}')
        plt.legend()
        plt.grid(True)
        plt.show()
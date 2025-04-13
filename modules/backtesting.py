#!/usr/bin/env python3
"""
Backtesting module for the Qlib-based quantitative analysis project.
This module provides classes and functions for backtesting trading strategies
based on forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modules.storage import StorageManager


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, forecasts: List[float]) -> pd.DataFrame:
        """
        Generate trading signals based on historical data and forecasts.
        
        Args:
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        pass


class SimpleMomentumStrategy(TradingStrategy):
    """Simple momentum-based trading strategy."""
    
    def __init__(self, threshold: float = 0.01):
        """
        Initialize the strategy.
        
        Args:
            threshold (float): Price change threshold for generating signals
        """
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, forecasts: List[float]) -> pd.DataFrame:
        """
        Generate trading signals based on historical data and forecasts.
        
        Args:
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy of the data
        signals = data.copy()
        
        # Calculate daily returns
        signals['returns'] = signals['Close'].pct_change()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Generate signals based on momentum
        signals['momentum'] = signals['returns'].rolling(window=5).mean()
        signals['signal'] = np.where(signals['momentum'] > self.threshold, 1, 0)
        signals['signal'] = np.where(signals['momentum'] < -self.threshold, -1, signals['signal'])
        
        # Add forecast signals
        if forecasts and len(forecasts) > 0:
            last_price = signals['Close'].iloc[-1]
            forecast_returns = [(price / last_price) - 1 for price in forecasts]
            
            # Debug logging
            print(f"Last price: {last_price}")
            print(f"Forecast returns: {forecast_returns}")
            
            # If the average forecast return is above threshold, set signal to 1 (buy/hold)
            # If it's below negative threshold, set signal to -1 (sell/short)
            avg_forecast_return = sum(forecast_returns) / len(forecast_returns)
            print(f"Average forecast return: {avg_forecast_return}, type: {type(avg_forecast_return)}")
            
            if avg_forecast_return > self.threshold:
                signals.loc[signals.index[-1], 'signal'] = 1
            elif avg_forecast_return < -self.threshold:
                signals.loc[signals.index[-1], 'signal'] = -1
        
        return signals


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        """
        Initialize the strategy.
        
        Args:
            window (int): Lookback window for calculating mean and standard deviation
            std_dev (float): Number of standard deviations for generating signals
        """
        self.window = window
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame, forecasts: List[float]) -> pd.DataFrame:
        """
        Generate trading signals based on historical data and forecasts.
        
        Args:
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy of the data
        signals = data.copy()
        
        # Calculate rolling mean and standard deviation
        signals['rolling_mean'] = signals['Close'].rolling(window=self.window).mean()
        signals['rolling_std'] = signals['Close'].rolling(window=self.window).std()
        
        # Calculate z-score (avoiding division by zero)
        signals['z_score'] = np.where(
            signals['rolling_std'] != 0,
            (signals['Close'] - signals['rolling_mean']) / signals['rolling_std'],
            0
        )
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Generate signals based on z-score
        signals['signal'] = np.where(signals['z_score'] < -self.std_dev, 1, 0)  # Buy when price is below mean
        signals['signal'] = np.where(signals['z_score'] > self.std_dev, -1, signals['signal'])  # Sell when price is above mean
        
        # Add forecast signals
        if forecasts and len(forecasts) > 0:
            last_price = signals['Close'].iloc[-1]
            last_mean = signals['rolling_mean'].iloc[-1]
            last_std = signals['rolling_std'].iloc[-1]
            
            # Calculate z-score for forecasts
            # Calculate z-scores safely
            forecast_z_scores = [(price - last_mean) / last_std if last_std != 0 else 0 for price in forecasts]
            avg_z_score = sum(forecast_z_scores) / len(forecast_z_scores) if forecast_z_scores else 0
            
            # Update signal based on forecast z-score
            if avg_z_score < -self.std_dev:
                signals.loc[signals.index[-1], 'signal'] = 1
            elif avg_z_score > self.std_dev:
                signals.loc[signals.index[-1], 'signal'] = -1
        
        return signals


class ForecastBasedStrategy(TradingStrategy):
    """Trading strategy based purely on forecasts."""
    
    def __init__(self, threshold: float = 0.01):
        """
        Initialize the strategy.
        
        Args:
            threshold (float): Price change threshold for generating signals
        """
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, forecasts: List[float]) -> pd.DataFrame:
        """
        Generate trading signals based on historical data and forecasts.
        
        Args:
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Create a copy of the data
        signals = data.copy()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Generate signals based on forecasts
        if forecasts and len(forecasts) > 0:
            last_price = signals['Close'].iloc[-1]
            
            # Calculate forecast returns
            forecast_returns = [(price / last_price) - 1 for price in forecasts]
            
            # Use short-term and long-term forecast returns
            short_term_return = forecast_returns[0] if len(forecast_returns) > 0 else 0
            long_term_return = forecast_returns[-1] if len(forecast_returns) > 0 else 0
            
            # Generate signal based on forecast returns
            if short_term_return > self.threshold and long_term_return > self.threshold:
                signals.loc[:, 'signal'] = 1  # Strong buy signal
            elif short_term_return > self.threshold:
                signals.loc[:, 'signal'] = 0.5  # Weak buy signal
            elif short_term_return < -self.threshold and long_term_return < -self.threshold:
                signals.loc[:, 'signal'] = -1  # Strong sell signal
            elif short_term_return < -self.threshold:
                signals.loc[:, 'signal'] = -0.5  # Weak sell signal
        
        return signals


class BacktestResult:
    """Class for storing and analyzing backtest results."""
    
    def __init__(self, symbol: str, signals: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize backtest result.
        
        Args:
            symbol (str): Stock symbol
            signals (pd.DataFrame): DataFrame with trading signals
            initial_capital (float): Initial capital for the backtest
        """
        self.symbol = symbol
        self.signals = signals
        self.initial_capital = initial_capital
        
        # Calculate positions and portfolio value
        self._calculate_positions()
        self._calculate_portfolio()
        self._calculate_metrics()
    
    def _calculate_positions(self):
        """Calculate positions based on signals."""
        # Create position column
        self.signals['position'] = self.signals['signal'].shift(1)
        self.signals.loc[self.signals.index[0], 'position'] = 0
        
        # Calculate position changes
        self.signals['position_diff'] = self.signals['position'].diff()
    
    def _calculate_portfolio(self):
        """Calculate portfolio value based on positions."""
        # Calculate holdings and cash
        self.signals['holdings'] = self.signals['position'] * self.signals['Close']
        self.signals['cash'] = self.initial_capital - (self.signals['position_diff'] * self.signals['Close']).cumsum()
        
        # Calculate portfolio value
        self.signals['portfolio'] = self.signals['holdings'] + self.signals['cash']
        
        # Calculate daily returns
        self.signals['portfolio_returns'] = self.signals['portfolio'].pct_change()
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        # Total return
        self.total_return = (self.signals['portfolio'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = (self.signals.index[-1] - self.signals.index[0]).days
        self.annual_return = (1 + self.total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Maximum drawdown
        self.signals['drawdown'] = (self.signals['portfolio'] / self.signals['portfolio'].cummax()) - 1
        self.max_drawdown = self.signals['drawdown'].min()
        
        # Sharpe ratio (assuming risk-free rate of 0)
        std = self.signals['portfolio_returns'].std()
        mean_return = self.signals['portfolio_returns'].mean()
        self.sharpe_ratio = (mean_return / std) * np.sqrt(252) if std != 0 and not np.isnan(std) else 0
        
        # Win rate
        trades = self.signals[self.signals['position_diff'] != 0]
        if len(trades) > 0:
            trade_returns = trades['portfolio'].pct_change(fill_method=None)
            self.win_rate = (trade_returns > 0).sum() / len(trade_returns)
        else:
            self.win_rate = 0
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        return {
            'symbol': self.symbol,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate
        }
    
    def plot_portfolio(self):
        """Plot portfolio value over time."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.signals.index, self.signals['portfolio'])
            plt.title(f'Portfolio Value Over Time - {self.symbol}')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            
            return plt.gcf()
        except ImportError:
            print("Matplotlib is required for plotting")
            return None


class BacktestingManager:
    """
    Manager for backtesting operations.
    Handles strategy evaluation and optimization.
    """
    
    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize backtesting manager.
        
        Args:
            storage_manager (StorageManager, optional): Storage manager for parameter persistence
        """
        self.storage_manager = storage_manager if storage_manager else StorageManager()
        
        # Available strategies
        self.strategies = {
            'momentum': SimpleMomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'forecast_based': ForecastBasedStrategy
        }
    
    def run_backtest(self, symbol: str, data: pd.DataFrame, forecasts: List[float], 
                     strategy_name: str = 'momentum', strategy_params: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """
        Run a backtest for a single stock.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            strategy_name (str): Name of the strategy to use
            strategy_params (Dict[str, Any], optional): Parameters for the strategy
            
        Returns:
            BacktestResult: Backtest result
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Create strategy
        strategy_class = self.strategies[strategy_name]
        strategy = strategy_class(**(strategy_params or {}))
        
        # Generate signals
        signals = strategy.generate_signals(data, forecasts)
        
        # Create backtest result
        result = BacktestResult(symbol, signals)
        
        return result
    
    def optimize_strategy(self, symbol: str, data: pd.DataFrame, forecasts: List[float],
                          strategy_name: str = 'momentum', param_grid: Optional[Dict[str, List[Any]]] = None,
                          optimization_metric: str = 'sharpe_ratio', test_period: float = 0.3) -> Tuple[Dict[str, Any], BacktestResult, Dict[str, Any]]:
        """
        Optimize strategy parameters using walk-forward validation.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            strategy_name (str): Name of the strategy to optimize
            param_grid (Dict[str, List[Any]], optional): Parameter grid for optimization
            optimization_metric (str): Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            test_period (float): Fraction of data to use for testing (0.0 to 1.0)
            
        Returns:
            Tuple[Dict[str, Any], BacktestResult, Dict[str, Any]]: Best parameters, backtest result, and optimization stats
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Use default parameter grid if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid(strategy_name)
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Split data into training and testing sets
        split_idx = int(len(data) * (1 - test_period))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        best_params = None
        best_result = None
        best_metric = -float('inf')  # Assuming higher is better for the metric
        all_results = []
        
        print(f"Optimizing {strategy_name} strategy for {symbol} with {len(param_combinations)} parameter combinations")
        
        # Test each parameter combination
        for params in param_combinations:
            # Train on training set
            train_result = self.run_backtest(symbol, train_data, forecasts, strategy_name, params)
            
            # Test on test set
            test_result = self.run_backtest(symbol, test_data, forecasts, strategy_name, params)
            
            # Use specified optimization metric
            if optimization_metric == 'sharpe_ratio':
                metric = test_result.sharpe_ratio
            elif optimization_metric == 'total_return':
                metric = test_result.total_return
            elif optimization_metric == 'annual_return':
                metric = test_result.annual_return
            elif optimization_metric == 'win_rate':
                metric = test_result.win_rate
            else:
                # Default to Sharpe ratio
                metric = test_result.sharpe_ratio
            
            # Store results for analysis
            result_data = {
                'params': params,
                'train_metrics': train_result.get_metrics(),
                'test_metrics': test_result.get_metrics(),
                'optimization_metric': metric
            }
            all_results.append(result_data)
            
            if metric > best_metric:
                best_metric = metric
                best_params = params
                best_result = self.run_backtest(symbol, data, forecasts, strategy_name, params)  # Run on full dataset
        
        # Calculate optimization statistics
        optimization_stats = {
            'num_combinations': len(param_combinations),
            'best_metric_value': best_metric,
            'best_params': best_params,
            'parameter_impact': self._analyze_parameter_impact(all_results, param_grid),
            'train_test_correlation': self._calculate_train_test_correlation(all_results, optimization_metric)
        }
        
        # Save best parameters if storage manager is available
        if self.storage_manager:
            self.storage_manager.save_model_params(symbol, f"backtest_{strategy_name}", best_params)
        
        return best_params, best_result, optimization_stats
    
    def _get_default_param_grid(self, strategy_name: str) -> Dict[str, List[Any]]:
        """
        Get default parameter grid for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            Dict[str, List[Any]]: Default parameter grid
        """
        if strategy_name == 'momentum':
            return {
                'threshold': [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
            }
        elif strategy_name == 'mean_reversion':
            return {
                'window': [5, 10, 15, 20, 25, 30],
                'std_dev': [1.0, 1.5, 2.0, 2.5, 3.0]
            }
        elif strategy_name == 'forecast_based':
            return {
                'threshold': [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
            }
        else:
            return {}
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from the parameter grid.
        
        Args:
            param_grid (Dict[str, List[Any]]): Parameter grid
            
        Returns:
            List[Dict[str, Any]]: List of parameter combinations
        """
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _analyze_parameter_impact(self, results: List[Dict[str, Any]], param_grid: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze the impact of each parameter on the optimization metric.
        
        Args:
            results: List of results from parameter optimization
            param_grid: Parameter grid used for optimization
            
        Returns:
            Dict with parameter impact analysis
        """
        impact_analysis = {}
        
        # For each parameter
        for param_name, param_values in param_grid.items():
            param_impact = {}
            
            # For each value of this parameter
            for value in param_values:
                # Get results for this parameter value
                value_results = [r for r in results if r['params'].get(param_name) == value]
                
                if value_results:
                    # Calculate average metric for this parameter value
                    avg_metric = sum(r['optimization_metric'] for r in value_results) / len(value_results)
                    param_impact[str(value)] = avg_metric
            
            impact_analysis[param_name] = param_impact
        
        return impact_analysis
    
    def _calculate_train_test_correlation(self, results: List[Dict[str, Any]], metric: str) -> float:
        """
        Calculate correlation between training and testing performance.
        
        Args:
            results: List of results from parameter optimization
            metric: Metric to analyze
            
        Returns:
            Correlation coefficient
        """
        if not results:
            return 0.0
        
        try:
            train_metrics = [r['train_metrics'].get(metric, 0) if metric in r['train_metrics']
                            else r['train_metrics'].get('sharpe_ratio', 0) for r in results]
            test_metrics = [r['test_metrics'].get(metric, 0) if metric in r['test_metrics']
                           else r['test_metrics'].get('sharpe_ratio', 0) for r in results]
            
            if len(train_metrics) > 1:
                return np.corrcoef(train_metrics, test_metrics)[0, 1]
            else:
                return 0.0
        except:
            return 0.0
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def evaluate_forecasts(self, symbol: str, data: pd.DataFrame, forecasts: List[float]) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using historical data.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Need at least 10 data points for evaluation
        if len(data) < 10 or not forecasts:
            return {}
        
        # Use the last 10 days as the test set
        test_data = data.iloc[-10:].copy()
        actual_prices = test_data['Close'].values
        
        # Use only as many forecast values as we have actual prices
        forecast_prices = forecasts[:len(actual_prices)]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_prices, forecast_prices)
        mse = mean_squared_error(actual_prices, forecast_prices)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Calculate MAPE safely (avoiding division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual_prices - forecast_prices) / np.where(actual_prices != 0, actual_prices, np.inf))) * 100
        
        # Calculate R-squared
        r2 = r2_score(actual_prices, forecast_prices)
        
        return {
            'symbol': symbol,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }


# For backward compatibility with existing code
def run_backtest(data_dict, forecast=None):
    """
    Run a simple backtesting algorithm on the provided stock data.
    This function is maintained for backward compatibility.
    """
    results = {}
    
    manager = BacktestingManager()
    
    for symbol, data in data_dict.items():
        if not data.empty:
            # Get forecast for this symbol
            symbol_forecast = forecast.get(symbol, []) if forecast else []
            
            # Run backtest with default strategy
            result = manager.run_backtest(symbol, data, symbol_forecast)
            
            # Get metrics
            metrics = result.get_metrics()
            results[symbol] = metrics
        else:
            results[symbol] = {"return": 0.0}
    
    return results
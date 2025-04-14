#!/usr/bin/env python3
"""
Backtesting module for the Qlib-based quantitative analysis project.
This module provides classes and functions for backtesting trading strategies
using Qlib's backtesting framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod

from modules.storage import StorageManager


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, stop_loss: float = 0.02, take_profit: float = 0.03,
                 max_position_size: float = 1.0, volatility_scaling: bool = True):
        """
        Initialize strategy with risk management parameters.
        
        Args:
            stop_loss (float): Stop loss percentage (default: 2%)
            take_profit (float): Take profit percentage (default: 3%)
            max_position_size (float): Maximum position size as fraction of portfolio (default: 100%)
            volatility_scaling (bool): Whether to scale position size by volatility (default: True)
        """
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_size = max_position_size
        self.volatility_scaling = volatility_scaling
    
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
    
    def _calculate_position_size(self, data: pd.DataFrame) -> float:
        """Calculate position size based on volatility if enabled."""
        if not self.volatility_scaling:
            return self.max_position_size
            
        # Calculate volatility using 20-day rolling standard deviation
        volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        target_volatility = 0.01  # Target 1% daily volatility
        
        # Scale position size inversely with volatility
        volatility_value = volatility.iloc[0] if isinstance(volatility, pd.Series) else volatility
        if np.isclose(volatility_value, 0):  # Safe comparison with zero
            position_size = self.max_position_size
        else:
            scale = target_volatility / volatility_value
            position_size = min(float(self.max_position_size),
                              float(self.max_position_size) * scale)
        return position_size


class SimpleMomentumStrategy(TradingStrategy):
    """Simple momentum-based trading strategy."""
    
    def __init__(self, threshold: float = 0.01, stop_loss: float = 0.02,
                 take_profit: float = 0.03, max_position_size: float = 1.0,
                 volatility_scaling: bool = True):
        """
        Initialize the strategy.
        
        Args:
            threshold (float): Price change threshold for generating signals
            stop_loss (float): Stop loss percentage (default: 2%)
            take_profit (float): Take profit percentage (default: 3%)
            max_position_size (float): Maximum position size as fraction of portfolio
            volatility_scaling (bool): Whether to scale position size by volatility
        """
        super().__init__(stop_loss, take_profit, max_position_size, volatility_scaling)
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
        
        # Initialize signal and position size columns
        signals['signal'] = 0
        signals['position_size'] = 0
        
        # Generate signals based on momentum
        signals['momentum'] = signals['returns'].rolling(window=5).mean()
        signals['signal'] = np.where(signals['momentum'] > self.threshold, 1, 0)
        signals['signal'] = np.where(signals['momentum'] < -self.threshold, -1, signals['signal'])
        
        # Calculate position size based on volatility
        position_size = self._calculate_position_size(data)
        
        # Apply position sizing to signals
        signals['position_size'] = signals['signal'] * position_size
        
        # Implement stop-loss and take-profit
        entry_price = None
        position = 0
        
        for i in range(1, len(signals)):
            current_price = signals['Close'].iloc[i]
            
            # If we have an open position, check stop-loss and take-profit
            if position != 0 and entry_price is not None:
                returns = (current_price / entry_price - 1) * np.sign(position)
                
                # Check stop-loss
                current_returns = returns.iloc[0] if isinstance(returns, pd.Series) else returns
                if current_returns < -self.stop_loss:
                    signals.loc[signals.index[i], 'signal'] = 0
                    signals.loc[signals.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                    continue
                
                # Check take-profit
                if current_returns > self.take_profit:  # Use current_returns from previous check
                    signals.loc[signals.index[i], 'signal'] = 0
                    signals.loc[signals.index[i], 'position_size'] = 0
                    position = 0
                    entry_price = None
                    continue
            
            # Update position tracking if signal changes
            current_signal = signals['signal'].iloc[i]
            if current_signal != 0 and current_signal != position:
                position = current_signal
                entry_price = current_price
        
        # Add forecast signals with position sizing
        if forecasts and len(forecasts) > 0:
            last_price = signals['Close'].iloc[-1]  # Already a scalar from .iloc[-1]
            forecast_returns = [
                (price.iloc[0] / last_price if isinstance(price, pd.Series) else price / last_price) - 1
                for price in forecasts
            ]
            avg_forecast_return = sum(forecast_returns) / len(forecast_returns)
            threshold = float(self.threshold) if isinstance(self.threshold, pd.Series) else float(self.threshold)
            
            # Get scalar values for comparison
            forecast_return_value = float(avg_forecast_return.iloc[0]) if isinstance(avg_forecast_return, pd.Series) else float(avg_forecast_return)
            threshold_value = float(threshold.iloc[0]) if isinstance(threshold, pd.Series) else float(threshold)
            position_size_value = float(position_size.iloc[0]) if isinstance(position_size, pd.Series) else float(position_size)
            
            # Compare scalar values
            if forecast_return_value > threshold_value:
                signals.loc[signals.index[-1], 'signal'] = 1.0
                signals.loc[signals.index[-1], 'position_size'] = position_size_value
            elif forecast_return_value < -threshold_value:
                signals.loc[signals.index[-1], 'signal'] = -1.0
                signals.loc[signals.index[-1], 'position_size'] = -position_size_value
        
        return signals


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, window: int = 20, std_dev: float = 2.0, stop_loss: float = 0.02,
                 take_profit: float = 0.03, max_position_size: float = 1.0,
                 volatility_scaling: bool = True):
        """
        Initialize the strategy.
        
        Args:
            window (int): Lookback window for calculating mean and standard deviation
            std_dev (float): Number of standard deviations for generating signals
            stop_loss (float): Stop loss percentage (default: 2%)
            take_profit (float): Take profit percentage (default: 3%)
            max_position_size (float): Maximum position size as fraction of portfolio
            volatility_scaling (bool): Whether to scale position size by volatility
        """
        super().__init__(stop_loss, take_profit, max_position_size, volatility_scaling)
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
        # Ensure all inputs are Series with matching dimensions
        close_series = signals['Close'] if isinstance(signals['Close'], pd.Series) else signals['Close'].iloc[:, 0]
        mean_series = signals['rolling_mean'] if isinstance(signals['rolling_mean'], pd.Series) else signals['rolling_mean'].iloc[:, 0]
        std_series = signals['rolling_std'] if isinstance(signals['rolling_std'], pd.Series) else signals['rolling_std'].iloc[:, 0]
        
        # Calculate z-score with explicit Series operations
        signals['z_score'] = np.where(
            std_series != 0,
            (close_series - mean_series) / std_series,
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
    
    def __init__(self, threshold: float = 0.01, stop_loss: float = 0.02,
                 take_profit: float = 0.03, max_position_size: float = 1.0,
                 volatility_scaling: bool = True):
        """
        Initialize the strategy.
        
        Args:
            threshold (float): Price change threshold for generating signals
            stop_loss (float): Stop loss percentage (default: 2%)
            take_profit (float): Take profit percentage (default: 3%)
            max_position_size (float): Maximum position size as fraction of portfolio
            volatility_scaling (bool): Whether to scale position size by volatility
        """
        super().__init__(stop_loss, take_profit, max_position_size, volatility_scaling)
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
            
            # Convert values to scalar for comparison
            short_term_val = short_term_return.item() if isinstance(short_term_return, pd.Series) else short_term_return
            long_term_val = long_term_return.item() if isinstance(long_term_return, pd.Series) else long_term_return
            threshold_val = self.threshold.item() if isinstance(self.threshold, pd.Series) else self.threshold
            
            # Generate signal based on forecast returns using scalar values
            if short_term_val > threshold_val and long_term_val > threshold_val:
                signals.loc[:, 'signal'] = 1  # Strong buy signal
            elif short_term_val > threshold_val:
                signals.loc[:, 'signal'] = 0.5  # Weak buy signal
            elif short_term_val < -threshold_val and long_term_val < -threshold_val:
                signals.loc[:, 'signal'] = -1  # Strong sell signal
            elif short_term_val < -threshold_val:
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
        # Calculate holdings and cash (using basic pandas operations)
        try:
            # Ensure we are working with Series
            if isinstance(self.signals['Close'], pd.DataFrame):
                close_values = self.signals['Close'].iloc[:, 0]  # Take first column
            else:
                close_values = self.signals['Close']
            
            # Convert numeric columns and verify they are Series
            position_values = pd.Series(self.signals['position']).astype(float)
            position_diff_values = pd.Series(self.signals['position_diff']).astype(float)
            close_values = pd.Series(close_values).astype(float)
            
            # Calculate values
            self.signals['holdings'] = position_values * close_values
            self.signals['trades_value'] = position_diff_values * close_values
            
        except Exception as e:
            print(f"Error in portfolio calculation: {e}")
            print(f"Data types: position={type(self.signals['position'])}, "
                  f"Close={type(self.signals['Close'])}")
        
        trades_value = self.signals['trades_value']
        self.signals['cash'] = self.initial_capital - trades_value.cumsum()
        
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
        
        # Calculate drawdown duration
        drawdown_periods = self.signals[self.signals['drawdown'] < 0].groupby(
            (self.signals['drawdown'] >= 0).cumsum()
        )
        self.max_drawdown_duration = max([len(g) for _, g in drawdown_periods]) if drawdown_periods else 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        std = self.signals['portfolio_returns'].std()
        mean_return = self.signals['portfolio_returns'].mean()
        self.sharpe_ratio = (mean_return / std) * np.sqrt(252) if std != 0 and not np.isnan(std) else 0
        
        # Sortino ratio (downside risk only)
        downside_returns = self.signals['portfolio_returns'][self.signals['portfolio_returns'] < 0]
        downside_std = downside_returns.std()
        self.sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std != 0 and not np.isnan(downside_std) else 0
        
        # Calculate trade metrics
        trades = self.signals[self.signals['position_diff'] != 0]
        if len(trades) > 0:
            trade_returns = trades['portfolio'].pct_change(fill_method=None)
            self.win_rate = (trade_returns > 0).sum() / len(trade_returns)
            self.avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
            self.avg_loss = trade_returns[trade_returns < 0].mean() if len(trade_returns[trade_returns < 0]) > 0 else 0
            self.profit_factor = abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()) if len(trade_returns[trade_returns < 0]) > 0 else float('inf')
        else:
            self.win_rate = 0
            self.avg_win = 0
            self.avg_loss = 0
            self.profit_factor = 0
        
        # Calculate risk-adjusted returns
        self.calmar_ratio = abs(self.annual_return / self.max_drawdown) if self.max_drawdown != 0 else float('inf')
    
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
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor
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


class QlibStrategyAdapter:
    """Adapter class to convert our strategies to Qlib-compatible format."""
    
    def __init__(self, strategy_cls, **kwargs):
        self.strategy = strategy_cls(**kwargs)
        
    def generate_signals(self, data):
        """Convert strategy signals to Qlib format."""
        signals = self.strategy.generate_signals(data)
        return {
            'signal': signals['signal'],
            'position': signals['position_size']
        }

class BacktestingManager:
    """
    Manager for backtesting operations using Qlib's framework.
    """
    
    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize backtesting manager with Qlib integration.
        
        Args:
            storage_manager (StorageManager, optional): Storage manager for parameter persistence
        """
        self.storage_manager = storage_manager if storage_manager else StorageManager()
        
        # Available strategies with custom implementation
        self.strategies = {
            'momentum': lambda **kwargs: SimpleMomentumStrategy(**kwargs),
            'mean_reversion': lambda **kwargs: MeanReversionStrategy(**kwargs),
            'forecast_based': lambda **kwargs: ForecastBasedStrategy(**kwargs)
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
                           optimization_metric: str = 'sharpe_ratio', test_period: float = 0.3) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Optimize strategy parameters using Qlib's backtesting framework.
        
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
        total_combinations = len(param_combinations)
        
        print(f"\nOptimizing {strategy_name} strategy for {symbol} with {total_combinations} parameter combinations")
        print("Training set size:", len(train_data), "Testing set size:", len(test_data))
        
        # Test each parameter combination with progress tracking
        for i, params in enumerate(param_combinations, 1):
            # Print progress every 5 combinations
            if i % 5 == 0:
                print(f"Progress: {i}/{total_combinations} ({(i/total_combinations)*100:.1f}%)")
                if best_metric != -float('inf'):
                    print(f"Best {optimization_metric} so far: {best_metric:.4f}")

        total_combinations = len(param_combinations)
        print(f"\nOptimizing {strategy_name} strategy for {symbol} with {total_combinations} parameter combinations")
        print("Training set size:", len(train_data), "Testing set size:", len(test_data))
        
        print(f"Optimizing {strategy_name} strategy for {symbol} with {len(param_combinations)} parameter combinations")
        
        # Test each parameter combination
        for params in param_combinations:
            # Run backtest on training data
            train_result = self.run_backtest(symbol, train_data, forecasts, strategy_name, params)
            test_result = self.run_backtest(symbol, test_data, forecasts, strategy_name, params)
            
            # Get the specified metric from test results
            if optimization_metric == 'sharpe_ratio':
                metric = test_result.sharpe_ratio
            elif optimization_metric == 'total_return':
                metric = test_result.total_return
            elif optimization_metric == 'annual_return':
                metric = test_result.annual_return
            elif optimization_metric == 'max_drawdown':
                metric = abs(test_result.max_drawdown)  # Convert to positive for optimization
            else:
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
        # Common risk management parameters for all strategies
        risk_params = {
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.02, 0.03, 0.04],
            'max_position_size': [0.5, 1.0],
            'volatility_scaling': [True]
        }
        
        strategy_specific_params = {}
        if strategy_name == 'momentum':
            strategy_specific_params = {
                'threshold': [0.01, 0.02, 0.03]
            }
        elif strategy_name == 'mean_reversion':
            strategy_specific_params = {
                'window': [5, 10, 15, 20, 25, 30],
                'std_dev': [1.0, 1.5, 2.0, 2.5, 3.0]
            }
        elif strategy_name == 'forecast_based':
            strategy_specific_params = {
                'threshold': [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
            }
            
        # Combine strategy-specific parameters with risk management parameters
        return {**strategy_specific_params, **risk_params}
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
        Evaluate forecast accuracy using standard metrics.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical stock data
            forecasts (List[float]): Forecast values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if len(data) < 10 or not forecasts:
            return {}
        
        # Get actual values for comparison
        actual_values = data['Close'].iloc[-len(forecasts):].values
        forecast_values = np.array(forecasts[:len(actual_values)])
        
        # Calculate error metrics
        mse = np.mean((actual_values - forecast_values) ** 2)
        mae = np.mean(np.abs(actual_values - forecast_values))
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(actual_values))
        forecast_direction = np.sign(np.diff(forecast_values))
        directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
        
        # Calculate correlation
        correlation = np.corrcoef(actual_values, forecast_values)[0, 1]
        
        return {
            'symbol': symbol,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation
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
#!/usr/bin/env python3
"""
Continuous optimization module for the Qlib-based quantitative analysis project.
This module provides continuous parameter optimization through backtest-predict cycles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

from modules.storage import StorageManager
from modules.backtesting import BacktestingManager
from modules.forecasting import ForecastingManager
from modules.parameter_optimizer import StockParameterManager
from modules.market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


class OptimizationCycle:
    """Class to track a single optimization cycle."""
    
    def __init__(self, symbol: str, start_params: Dict):
        self.symbol = symbol
        self.start_params = start_params
        self.current_params = start_params.copy()
        self.best_params = start_params.copy()
        self.start_time = datetime.now()
        self.iterations = 0
        self.results = []
    
    def add_result(self, params: Dict, metrics: Dict):
        """Add optimization iteration result."""
        self.results.append({
            'params': params.copy(),
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best parameters if this result is better
        if self._is_better_result(metrics):
            self.best_params = params.copy()
    
    def _is_better_result(self, metrics: Dict) -> bool:
        """Compare new metrics with current best."""
        if not self.results:
            return True
            
        current_score = self._calculate_score(metrics)
        best_score = self._calculate_score(self.results[-1]['metrics'])
        
        return current_score > best_score
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall performance score."""
        return (
            metrics.get('accuracy', 0) * 0.3 +
            metrics.get('sharpe_ratio', 0) * 0.3 +
            metrics.get('win_rate', 0) * 0.2 +
            (1 - abs(metrics.get('max_drawdown', 1))) * 0.2
        )


class ContinuousOptimizer:
    """Manages continuous optimization of stock parameters through backtesting."""
    
    def __init__(self, 
                 parameter_manager: Optional[StockParameterManager] = None,
                 backtest_manager: Optional[BacktestingManager] = None,
                 forecast_manager: Optional[ForecastingManager] = None,
                 market_analyzer: Optional[MarketAnalyzer] = None):
        """Initialize continuous optimizer with all required components."""
        self.parameter_manager = parameter_manager or StockParameterManager()
        self.backtest_manager = backtest_manager or BacktestingManager()
        self.forecast_manager = forecast_manager or ForecastingManager()
        self.market_analyzer = market_analyzer or MarketAnalyzer()
        
        # Optimization settings
        self.optimization_window = 21  # trading days
        self.performance_threshold = 0.6  # 60% minimum performance
        self.max_iterations = 5
        self.min_improvement = 0.05  # 5% minimum improvement to continue
        
        # Current optimization state
        self.active_cycles = {}  # symbol -> OptimizationCycle
        
    def start_optimization(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Start a new optimization cycle for a symbol.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical price data
            
        Returns:
            Dict: Initial parameters
        """
        try:
            # Get current parameters
            current_params = self.parameter_manager.get_stock_parameters(symbol)
            
            # Create new optimization cycle
            self.active_cycles[symbol] = OptimizationCycle(symbol, current_params)
            
            # Run first optimization iteration
            return self.run_optimization_iteration(symbol, data)
            
        except Exception as e:
            logger.error(f"Failed to start optimization for {symbol}: {e}")
            return self.parameter_manager.get_stock_parameters(symbol)
    
    def run_optimization_iteration(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Run one iteration of the optimization cycle.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Historical price data
            
        Returns:
            Dict: Optimized parameters
        """
        if symbol not in self.active_cycles:
            return self.start_optimization(symbol, data)
        
        cycle = self.active_cycles[symbol]
        
        try:
            # Analyze market conditions
            market_conditions = self.market_analyzer.analyze_market(data)
            
            # Run backtest with current parameters
            backtest_result = self._run_backtest(symbol, data, cycle.current_params)
            
            # Generate forecasts
            forecast_result = self._generate_forecast(symbol, data, cycle.current_params)
            
            # Evaluate performance
            performance_metrics = self._evaluate_performance(backtest_result, forecast_result)
            
            # Update cycle with results
            cycle.add_result(cycle.current_params, performance_metrics)
            
            # Update parameters based on results
            new_params = self.parameter_manager.update_parameters(
                symbol, performance_metrics, market_conditions.to_dict())
            
            # Check if we should rollback
            if self.parameter_manager.should_rollback(symbol, performance_metrics):
                logger.info(f"Rolling back parameters for {symbol} due to poor performance")
                new_params = self.parameter_manager.rollback_parameters(symbol)
            
            cycle.current_params = new_params
            cycle.iterations += 1
            
            # Check if we should continue optimization
            if self._should_stop_optimization(cycle):
                logger.info(f"Stopping optimization for {symbol} after {cycle.iterations} iterations")
                self._finish_optimization(symbol)
                return cycle.best_params
            
            return new_params
            
        except Exception as e:
            logger.error(f"Optimization iteration failed for {symbol}: {e}")
            return cycle.current_params
    
    def _run_backtest(self, symbol: str, data: pd.DataFrame, parameters: Dict):
        """Run backtest with given parameters."""
        lookback = parameters['lookback_window']
        recent_data = data.iloc[-lookback:]
        
        return self.backtest_manager.run_backtest(
            symbol=symbol,
            data=recent_data,
            strategy_name='momentum',
            strategy_params={'threshold': 0.01}
        )
    
    def _generate_forecast(self, symbol: str, data: pd.DataFrame, parameters: Dict):
        """Generate forecasts with given parameters."""
        return self.forecast_manager.generate_forecast(
            symbol=symbol,
            data=data,
            model_type='ensemble'
        )
    
    def _evaluate_performance(self, backtest_result: Any, forecast_result: Dict) -> Dict:
        """Evaluate performance of current parameters."""
        metrics = {
            'accuracy': self._calculate_forecast_accuracy(forecast_result),
            'sharpe_ratio': getattr(backtest_result, 'sharpe_ratio', 0.0),
            'max_drawdown': getattr(backtest_result, 'max_drawdown', 0.0),
            'win_rate': getattr(backtest_result, 'win_rate', 0.0)
        }
        
        # Add volatility-adjusted metrics
        if hasattr(backtest_result, 'portfolio_returns'):
            returns = backtest_result.portfolio_returns
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = getattr(backtest_result, 'sortino_ratio', 0.0)
        
        return metrics
    
    def _calculate_forecast_accuracy(self, forecast_result: Dict) -> float:
        """Calculate forecast accuracy from results."""
        try:
            return forecast_result.get('directional_accuracy', 0.0) / 100.0
        except:
            return 0.0
    
    def _should_stop_optimization(self, cycle: OptimizationCycle) -> bool:
        """Determine if optimization should stop."""
        # Stop if we've reached max iterations
        if cycle.iterations >= self.max_iterations:
            return True
        
        # Stop if we've achieved good performance
        if cycle.results and cycle.results[-1]['metrics']['accuracy'] > self.performance_threshold:
            return True
        
        # Stop if we're not seeing enough improvement
        if len(cycle.results) >= 2:
            last_score = self._calculate_score(cycle.results[-1]['metrics'])
            prev_score = self._calculate_score(cycle.results[-2]['metrics'])
            if (last_score - prev_score) < self.min_improvement:
                return True
        
        return False
    
    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate overall performance score."""
        return (
            metrics.get('accuracy', 0) * 0.3 +
            metrics.get('sharpe_ratio', 0) * 0.3 +
            metrics.get('win_rate', 0) * 0.2 +
            (1 - abs(metrics.get('max_drawdown', 1))) * 0.2
        )
    
    def _finish_optimization(self, symbol: str):
        """Clean up after optimization cycle completes."""
        if symbol in self.active_cycles:
            cycle = self.active_cycles[symbol]
            
            # Save optimization history
            self._save_optimization_history(cycle)
            
            # Clean up
            del self.active_cycles[symbol]
    
    def _save_optimization_history(self, cycle: OptimizationCycle):
        """Save optimization history to storage."""
        try:
            history = {
                'symbol': cycle.symbol,
                'start_time': cycle.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'iterations': cycle.iterations,
                'start_params': cycle.start_params,
                'best_params': cycle.best_params,
                'results': cycle.results
            }
            
            # Save using parameter manager's storage
            self.parameter_manager.storage_manager.save_model_params(
                cycle.symbol,
                f'optimization_history_{cycle.start_time.strftime("%Y%m%d_%H%M%S")}',
                history
            )
        except Exception as e:
            logger.error(f"Failed to save optimization history for {cycle.symbol}: {e}")
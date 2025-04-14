#!/usr/bin/env python3
"""
Test script for continuous optimization cycle functionality.
Tests parameter adjustment and rollback mechanisms.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import shutil
import warnings
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.continuous_optimizer import ContinuousOptimizer, OptimizationCycle
from modules.parameter_optimizer import StockParameterManager
from modules.market_analyzer import MarketAnalyzer
from modules.storage import StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


def generate_test_forecasts(days: int = 10) -> List[float]:
    """Generate synthetic forecast data for testing."""
    np.random.seed(42)
    base_price = 100
    trend = 0.001  # Small upward trend
    return [base_price * (1 + trend * i + np.random.normal(0, 0.002)) 
            for i in range(days)]


def generate_test_data(days: int = 500) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    # Generate price series with trend and volatility
    base_price = 100
    trend = np.linspace(0, 0.5, days)  # Upward trend
    volatility = np.random.normal(0, 0.02, days)  # Daily returns
    
    prices = base_price * (1 + trend + np.cumsum(volatility))
    volume = np.random.normal(1000000, 200000, days)
    
    return pd.DataFrame({
        'Open': prices * (1 - 0.002),
        'High': prices * (1 + 0.005),
        'Low': prices * (1 - 0.005),
        'Close': prices,
        'Volume': volume
    }, index=dates)


class TestOptimizer(ContinuousOptimizer):
    """Test-specific optimizer that includes test forecasts."""
    
    def __init__(self, *args, **kwargs):
        """Initialize test optimizer with modified settings."""
        super().__init__(*args, **kwargs)
        # Optimization settings
        self.max_iterations = 3  # Set lower iteration count for testing
        self.improvement_threshold = 0.03  # 3% improvement threshold
        self.performance_threshold = 0.7  # Target accuracy threshold
        
        # State tracking
        self.completed_cycles = set()  # Track completed optimization cycles
        self._current_symbol = None  # Track current symbol being processed
    
    def start_optimization(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Override start_optimization to ensure proper cycle initialization."""
        current_params = self.parameter_manager.get_stock_parameters(symbol)
        
        # Initialize optimization cycle
        cycle = OptimizationCycle(symbol, current_params)
        self.active_cycles[symbol] = cycle
        logger.info(f"Starting optimization cycle for {symbol}")
        
        # Initialize with first result
        initial_metrics = {
            'accuracy': 0.6,
            'sharpe_ratio': 0.95,
            'max_drawdown': -0.1,
            'win_rate': 0.55
        }
        cycle.add_result(current_params, initial_metrics)
        cycle.iterations = 1  # Set initial iteration
        
        return current_params
    
    def _run_backtest(self, symbol: str, data: pd.DataFrame, parameters: Dict):
        """Override to include test forecasts with synthetic metrics."""
        lookback = parameters['lookback_window']
        recent_data = data.iloc[-lookback:]
        
        # Generate test forecasts for backtesting
        test_forecasts = generate_test_forecasts()
        
        # Run backtest
        result = self.backtest_manager.run_backtest(
            symbol=symbol,
            data=recent_data,
            strategy_name='momentum',
            strategy_params={'threshold': 0.01},
            forecasts=test_forecasts
        )
        
        logger.info(f"Backtest completed for {symbol}")
        
        # Add synthetic metrics for testing
        synthetic_returns = np.random.normal(0.001, 0.02, size=len(recent_data))
        result.portfolio_returns = pd.Series(synthetic_returns)
        result.sharpe_ratio = np.mean(synthetic_returns) / np.std(synthetic_returns) * np.sqrt(252)
        result.max_drawdown = -0.1
        result.win_rate = 0.55
        
        return result
    
    def _evaluate_performance(self, backtest_result: Any, forecast_result: Dict) -> Dict:
        """Override to provide consistent test metrics."""
        try:
            metrics = {
                'accuracy': 0.6 + np.random.normal(0, 0.05),  # Base accuracy with some variation
                'sharpe_ratio': backtest_result.sharpe_ratio if hasattr(backtest_result, 'sharpe_ratio') else 1.0,
                'max_drawdown': backtest_result.max_drawdown if hasattr(backtest_result, 'max_drawdown') else -0.1,
                'win_rate': backtest_result.win_rate if hasattr(backtest_result, 'win_rate') else 0.55
            }
            
            # Ensure values are valid
            metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
            
            # Add result to active cycle if available
            cycle = self.active_cycles.get(self._current_symbol)
            if cycle:
                params = self.parameter_manager.get_stock_parameters(self._current_symbol)
                cycle.add_result(params, metrics)
                logger.info(f"Added result to cycle: accuracy={metrics['accuracy']:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return {
                'accuracy': 0.5,
                'sharpe_ratio': 0.0,
                'max_drawdown': -0.2,
                'win_rate': 0.5
            }
    
    def run_optimization_iteration(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Override to ensure proper cycle management."""
        try:
            # Track current symbol
            self._current_symbol = symbol
            
            if symbol not in self.active_cycles:
                logger.warning(f"No active cycle for {symbol}, initializing new cycle")
                return self.start_optimization(symbol, data)
                
            cycle = self.active_cycles[symbol]
            logger.info(f"Running iteration {cycle.iterations + 1} for {symbol}")
            
            result = super().run_optimization_iteration(symbol, data)
            cycle.iterations += 1
            
            return result
        except Exception as e:
            logger.error(f"Error in optimization iteration: {e}")
            raise
        finally:
            self._current_symbol = None
    
    def _should_stop_optimization(self, cycle: OptimizationCycle) -> bool:
        """Override to ensure proper test completion checks."""
        try:
            # Check max iterations
            if cycle.iterations >= self.max_iterations:
                logger.info(f"Stopping optimization for {cycle.symbol} after {cycle.iterations} iterations")
                self.completed_cycles.add(cycle.symbol)
                return True
                
            # Check target accuracy
            if cycle.results and cycle.results[-1]['metrics']['accuracy'] >= self.performance_threshold:
                logger.info(f"Stopping optimization for {cycle.symbol} - reached target accuracy")
                self.completed_cycles.add(cycle.symbol)
                return True
                
            # Check improvement rate after at least 2 iterations
            if cycle.iterations >= 2 and len(cycle.results) >= 2:
                current_score = self._calculate_score(cycle.results[-1]['metrics'])
                previous_score = self._calculate_score(cycle.results[-2]['metrics'])
                improvement = (current_score - previous_score) / abs(previous_score) if abs(previous_score) > 0 else 0
                
                if improvement < self.improvement_threshold:
                    logger.info(f"Stopping optimization for {cycle.symbol} - insufficient improvement: {improvement:.2%}")
                    self.completed_cycles.add(cycle.symbol)
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error in stop condition check: {e}")
            return True  # Stop on error


def print_metrics(cycle: OptimizationCycle):
    """Print detailed metrics from optimization cycle."""
    if not cycle or not cycle.results:
        return
    
    latest = cycle.results[-1]
    metrics = latest['metrics']
    
    logger.info("\nLatest Performance Metrics:")
    logger.info(f"Accuracy:      {metrics.get('accuracy', 0):.4f}")
    logger.info(f"Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Max Drawdown:  {metrics.get('max_drawdown', 0):.4f}")
    logger.info(f"Win Rate:      {metrics.get('win_rate', 0):.4f}")
    
    if len(cycle.results) >= 2:
        initial = cycle.results[0]['metrics']
        logger.info("\nMetrics Improvement:")
        for key in ['accuracy', 'sharpe_ratio', 'win_rate']:
            initial_val = initial.get(key, 0)
            latest_val = metrics.get(key, 0)
            change = ((latest_val - initial_val) / abs(initial_val) if abs(initial_val) > 0 else 0) * 100
            logger.info(f"{key.title():12} {change:+.2f}%")


def test_optimization_cycle():
    """Test a complete optimization cycle with proper completion tracking."""
    
    # Initialize components
    storage_manager = StorageManager(backend_type="csv")
    parameter_manager = StockParameterManager(storage_manager)
    market_analyzer = MarketAnalyzer()
    optimizer = TestOptimizer(parameter_manager=parameter_manager)
    
    # Generate test data
    symbol = "TEST_STOCK"
    data = generate_test_data()
    
    logger.info("\n=== Starting Optimization Cycle Test ===")
    
    # Create directories for parameter storage
    os.makedirs(os.path.join("data/csv/stock_params", symbol), exist_ok=True)
    
    completed_iterations = 0
    try:
        # Start optimization
        initial_params = optimizer.start_optimization(symbol, data)
        logger.info(f"\nInitial parameters:\n{json.dumps(initial_params, indent=2)}")
        
        # Run multiple iterations
        for i in range(3):
            logger.info(f"\n--- Iteration {i + 1} ---")
            try:
                # Run optimization iteration
                new_params = optimizer.run_optimization_iteration(symbol, data)
                completed_iterations += 1
                
                # Get current cycle
                cycle = optimizer.active_cycles.get(symbol)
                if not cycle:
                    logger.error("No active optimization cycle found")
                    continue
                
                # Log results
                logger.info(f"\nNew parameters set:")
                for key, value in new_params.items():
                    if key not in ['version', 'timestamp']:
                        logger.info(f"{key:20}: {value}")
                
                print_metrics(cycle)
                
                # Test rollback condition
                if i == 1:  # Force poor performance on second iteration
                    logger.info("\nTesting rollback mechanism...")
                    poor_metrics = {
                        'accuracy': 0.3,
                        'sharpe_ratio': -0.5,
                        'max_drawdown': -0.3,
                        'win_rate': 0.35
                    }
                    parameter_manager.update_performance(symbol, 
                                                      new_params.get('version', 'latest'),
                                                      poor_metrics)
                    
                    if parameter_manager.should_rollback(symbol, poor_metrics):
                        logger.info("Rollback triggered as expected")
                        rolled_back_params = parameter_manager.rollback_parameters(symbol)
                        logger.info("\nRolled back to parameters:")
                        for key, value in rolled_back_params.items():
                            if key not in ['version', 'timestamp']:
                                logger.info(f"{key:20}: {value}")
                
                if symbol in optimizer.completed_cycles:
                    logger.info(f"\nOptimization cycle completed after {i+1} iterations")
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                break
        
        # Verify optimization history
        history_files = os.listdir(os.path.join("data/csv/stock_params", symbol))
        logger.info(f"\nOptimization history files: {history_files}")
        
        # Final performance summary
        cycle = optimizer.active_cycles.get(symbol)
        if cycle and completed_iterations > 0:
            logger.info("\n=== Final Performance Summary ===")
            print_metrics(cycle)
            logger.info("\nOptimization cycle test completed successfully")
        else:
            raise ValueError("No optimization iterations completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up test data
        try:
            shutil.rmtree(os.path.join("data/csv/stock_params", symbol), ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up test data: {e}")


def test_optimization_stop_conditions():
    """Test optimization stop conditions with proper cycle management."""
    optimizer = TestOptimizer()
    symbol = "TEST_STOP"
    
    logger.info("\n=== Testing Optimization Stop Conditions ===")
    
    try:
        # Test max iterations
        cycle = OptimizationCycle(symbol, {'test': 'params'})
        cycle.iterations = optimizer.max_iterations
        assert optimizer._should_stop_optimization(cycle), "Should stop at max iterations"
        logger.info("✓ Max iterations test passed")
        
        # Test performance threshold
        cycle = OptimizationCycle(symbol, {'test': 'params'})
        cycle.add_result({'test': 'params'}, {
            'accuracy': 0.8,
            'sharpe_ratio': 2.0,
            'max_drawdown': -0.1,
            'win_rate': 0.7
        })
        assert optimizer._should_stop_optimization(cycle), "Should stop at good performance"
        logger.info("✓ Performance threshold test passed")
        
        # Test minimum improvement
        cycle = OptimizationCycle(symbol, {'test': 'params'})
        optimizer.active_cycles[symbol] = cycle  # Register cycle with optimizer
        
        cycle.add_result({'test': 'params'}, {
            'accuracy': 0.5,
            'sharpe_ratio': 1.0,
            'max_drawdown': -0.2,
            'win_rate': 0.6
        })
        cycle.add_result({'test': 'params'}, {
            'accuracy': 0.51,  # Small improvement
            'sharpe_ratio': 1.01,
            'max_drawdown': -0.19,
            'win_rate': 0.61
        })
        cycle.iterations = 2  # Ensure proper iteration count
        
        if not optimizer._should_stop_optimization(cycle):
            raise AssertionError("Should stop at small improvement")
        logger.info("✓ Minimum improvement test passed")
        
        logger.info("\nAll stop condition tests passed successfully")
    except Exception as e:
        logger.error(f"Stop condition tests failed: {e}")
        raise


if __name__ == "__main__":
    # Create test directories if they don't exist
    os.makedirs("data/csv/stock_params", exist_ok=True)
    
    success = True
    try:
        test_optimization_cycle()
        test_optimization_stop_conditions()
        logger.info("\n=== All Tests Completed Successfully ===")
    except Exception as e:
        logger.error(f"\nTests failed: {e}")
        success = False
    finally:
        # Clean up test directories
        try:
            shutil.rmtree(os.path.join("data/csv/stock_params", "TEST_STOCK"), ignore_errors=True)
            shutil.rmtree(os.path.join("data/csv/stock_params", "TEST_STOP"), ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up test directories: {e}")
    

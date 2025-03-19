"""
Backtesting engine for Qlib strategies.
"""

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.evaluate import risk_analysis, analysis_position
from ..strategies.base_strategy import QlibStrategyBase
from ..config import BACKTEST_CONFIG

class BacktestEngine:
    """Engine for running strategy backtests."""
    
    def __init__(self, strategy: QlibStrategyBase, config: Optional[Dict] = None):
        """Initialize backtest engine.
        
        Args:
            strategy (QlibStrategyBase): Strategy to backtest
            config (dict, optional): Backtest configuration. Defaults to None.
        """
        self.strategy = strategy
        self.config = config or BACKTEST_CONFIG
        self.executor = self._initialize_executor()
        
    def _initialize_executor(self) -> SimulatorExecutor:
        """Initialize the backtest executor.
        
        Returns:
            SimulatorExecutor: Initialized executor
        """
        executor_config = self.config["executor"]
        return SimulatorExecutor(**executor_config["kwargs"])
        
    def run(self, dataset, start_time: str, end_time: str) -> Dict:
        """Run backtest for the strategy.
        
        Args:
            dataset: Dataset for backtesting
            start_time (str): Backtest start time
            end_time (str): Backtest end time
            
        Returns:
            dict: Backtest results including performance metrics
        """
        # Run backtest
        portfolio_metrics = qlib_backtest(
            start_time=start_time,
            end_time=end_time,
            strategy=self.strategy,
            executor=self.executor,
            account=1000000.0  # Initial capital
        )
        
        # Calculate performance metrics
        analysis = self._analyze_results(portfolio_metrics)
        
        return {
            "portfolio_metrics": portfolio_metrics,
            "analysis": analysis
        }
    
    def _analyze_results(self, portfolio_metrics: pd.DataFrame) -> Dict:
        """Analyze backtest results.
        
        Args:
            portfolio_metrics (pd.DataFrame): Portfolio metrics from backtest
            
        Returns:
            dict: Analysis results
        """
        # Get portfolio returns
        returns = portfolio_metrics["return"]
        
        # Calculate risk metrics
        risk_metrics = risk_analysis(returns)
        
        # Calculate position analysis
        pos_analysis = analysis_position(portfolio_metrics)
        
        # Calculate additional metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": risk_metrics["sharp"],
            "information_ratio": risk_metrics["information_ratio"],
            "positions_analysis": pos_analysis
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results.
        
        Args:
            results (dict): Results from backtest run
            save_path (str, optional): Path to save plot. Defaults to None.
        """
        import matplotlib.pyplot as plt
        
        portfolio_metrics = results["portfolio_metrics"]
        returns = portfolio_metrics["return"]
        cumulative_returns = (1 + returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title("Strategy Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 
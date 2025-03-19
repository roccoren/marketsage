"""
IEngine.py - Engine Interface

This module defines the standard interface for backtesting and live trading engines.
Following LEAN best practices, this ensures consistency across different engine implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IEngine(ABC):
    """
    Interface for backtesting and live trading engines.
    Following LEAN framework best practices.
    """
    
    @abstractmethod
    def Initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the engine with the given configuration.
        
        Args:
            config: Engine configuration
        """
        pass
    
    @abstractmethod
    def Run(self, algorithm: Any, settings: Any) -> Any:
        """
        Run the engine with the given algorithm and settings.
        
        Args:
            algorithm: The algorithm to run
            settings: The settings for the run
            
        Returns:
            The results of the run
        """
        pass
    
    @abstractmethod
    def StopEngine(self) -> None:
        """
        Stop the engine if it's running.
        """
        pass


class IBacktester(IEngine):
    """
    Interface specifically for backtesting engines.
    """
    
    @abstractmethod
    def RunBacktest(self, 
                  algorithm_name: str,
                  symbols: List[str],
                  start_date: Union[str, datetime],
                  end_date: Union[str, datetime],
                  initial_capital: float = 100000,
                  resolution: str = "Daily",
                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a backtest with the specified parameters.
        
        Args:
            algorithm_name: Name of the algorithm to use
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            resolution: Data resolution
            parameters: Additional parameters for the algorithm
            
        Returns:
            Dictionary containing backtest results
        """
        pass
    
    @abstractmethod
    def OptimizeParameters(self, 
                         algorithm_name: str,
                         symbols: List[str],
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime],
                         parameter_space: Dict[str, List[Any]],
                         initial_capital: float = 100000,
                         resolution: str = "Daily",
                         optimization_metric: str = "SharpeRatio",
                         max_concurrent_backtests: int = 4) -> Dict[str, Any]:
        """
        Optimize algorithm parameters using grid search or another optimization method.
        
        Args:
            algorithm_name: Name of the algorithm to optimize
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            parameter_space: Dictionary mapping parameter names to lists of values to try
            initial_capital: Initial capital for the backtest
            resolution: Data resolution
            optimization_metric: Metric to optimize (e.g., SharpeRatio, TotalReturn)
            max_concurrent_backtests: Maximum number of concurrent backtests to run
            
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    @abstractmethod
    def AnalyzeResults(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the results of a backtest.
        
        Args:
            backtest_result: The result of a backtest
            
        Returns:
            Dictionary containing analysis results
        """
        pass


class ILiveTrader(IEngine):
    """
    Interface specifically for live trading engines.
    """
    
    @abstractmethod
    def StartLiveTrading(self, 
                       algorithm_name: str,
                       symbols: List[str],
                       initial_capital: float = 100000,
                       parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Start live trading with the specified parameters.
        
        Args:
            algorithm_name: Name of the algorithm to use
            symbols: List of symbols to trade
            initial_capital: Initial capital for trading
            parameters: Additional parameters for the algorithm
            
        Returns:
            Deployment ID or similar identifier
        """
        pass
    
    @abstractmethod
    def StopLiveTrading(self, deployment_id: str) -> bool:
        """
        Stop live trading for the specified deployment.
        
        Args:
            deployment_id: Deployment ID or similar identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def GetLiveTradingStatus(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the status of a live trading deployment.
        
        Args:
            deployment_id: Deployment ID or similar identifier
            
        Returns:
            Dictionary containing status information
        """
        pass 
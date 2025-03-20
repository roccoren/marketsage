"""
Base strategy class for implementing trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        params: Optional[Dict] = None
    ):
        """
        Initialize the strategy
        
        Args:
            symbols: List of stock symbols to trade
            start_date: Start date for strategy
            end_date: End date for strategy
            params: Strategy parameters
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.params = params or {}
        self.positions: Dict[str, float] = {}
        self.signals: Dict[str, pd.Series] = {}
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals for each symbol
        
        Args:
            data: Dictionary mapping symbols to their DataFrames
            
        Returns:
            Dictionary mapping symbols to their signal series
        """
        pass
    
    @abstractmethod
    def calculate_positions(
        self,
        signals: Dict[str, pd.Series],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """
        Calculate position sizes based on signals
        
        Args:
            signals: Dictionary mapping symbols to their signal series
            data: Dictionary mapping symbols to their DataFrames
            
        Returns:
            Dictionary mapping symbols to their position series
        """
        pass
    
    def update_positions(
        self,
        positions: Dict[str, pd.Series],
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Update current positions based on new data
        
        Args:
            positions: Dictionary mapping symbols to their position series
            data: Dictionary mapping symbols to their DataFrames
        """
        for symbol in self.symbols:
            if symbol in positions and symbol in data:
                current_date = data[symbol].index[-1]
                self.positions[symbol] = positions[symbol].iloc[-1]
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current positions for all symbols
        
        Returns:
            Dictionary mapping symbols to their current position sizes
        """
        return self.positions
    
    def get_signals(self) -> Dict[str, pd.Series]:
        """
        Get trading signals for all symbols
        
        Returns:
            Dictionary mapping symbols to their signal series
        """
        return self.signals
    
    def calculate_returns(
        self,
        positions: Dict[str, pd.Series],
        data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        Calculate strategy returns
        
        Args:
            positions: Dictionary mapping symbols to their position series
            data: Dictionary mapping symbols to their DataFrames
            
        Returns:
            Series of strategy returns
        """
        # Initialize returns series with the first symbol's index
        first_symbol = list(data.keys())[0]
        returns = pd.Series(0.0, index=data[first_symbol].index)
        
        # Calculate weighted returns for each symbol
        for symbol in self.symbols:
            if symbol in positions and symbol in data:
                symbol_returns = data[symbol]['returns']
                symbol_positions = positions[symbol]
                returns += symbol_returns * symbol_positions
        
        return returns
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            returns: Series of strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate total return
        total_return = (1 + returns).prod() - 1
        
        # Calculate annual return
        years = (returns.index[-1] - returns.index[0]).days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        # Calculate profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        } 
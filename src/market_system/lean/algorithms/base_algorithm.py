"""
Base LEAN Algorithm

This module provides the base class for all LEAN-based trading algorithms.
It includes common functionality and utilities used across different strategies.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
from clr import AddReference
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")

from QuantConnect import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Parameters import *
from QuantConnect.Indicators import *

class BaseLeanAlgorithm(QCAlgorithm):
    """Base class for LEAN-based trading algorithms."""
    
    def Initialize(self) -> None:
        """Initialize the algorithm."""
        self.SetStartDate(2020, 1, 1)  # Set start date
        self.SetCash(100000)  # Set strategy cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        
        # Initialize strategy parameters
        self.symbols = []
        self.position_size = 0.1
        self.stop_loss = 0.02
        self.take_profit = 0.05
        
        # Risk management settings
        self.max_positions = 5
        self.max_drawdown = 0.15
        
        # Portfolio metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0
        }
    
    def OnData(self, data: TradeBar) -> None:
        """Process market data and execute trading logic.
        
        Args:
            data: Market data received
        """
        pass
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the algorithm's universe.
        
        Args:
            symbol: Symbol to add
        """
        equity = self.AddEquity(symbol, Resolution.Daily)
        self.symbols.append(equity.Symbol)
    
    def calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on portfolio value and risk parameters.
        
        Args:
            symbol: Symbol to calculate position size for
            
        Returns:
            Position size in number of shares
        """
        portfolio_value = self.Portfolio.TotalPortfolioValue
        price = self.Securities[symbol].Price
        
        # Calculate position size based on portfolio percentage
        position_value = portfolio_value * self.position_size
        return position_value / price
    
    def place_trade(self, symbol: str, direction: str, quantity: float) -> None:
        """Place a trade with stop loss and take profit orders.
        
        Args:
            symbol: Symbol to trade
            direction: 'long' or 'short'
            quantity: Number of shares to trade
        """
        if direction not in ['long', 'short']:
            self.Debug(f"Invalid trade direction: {direction}")
            return
            
        current_price = self.Securities[symbol].Price
        
        if direction == 'long':
            # Place long order with stop loss and take profit
            stop_price = current_price * (1 - self.stop_loss)
            take_profit_price = current_price * (1 + self.take_profit)
            
            self.MarketOrder(symbol, quantity)
            self.StopMarketOrder(symbol, -quantity, stop_price)
            self.LimitOrder(symbol, -quantity, take_profit_price)
            
        else:  # short
            # Place short order with stop loss and take profit
            stop_price = current_price * (1 + self.stop_loss)
            take_profit_price = current_price * (1 - self.take_profit)
            
            self.MarketOrder(symbol, -quantity)
            self.StopMarketOrder(symbol, quantity, stop_price)
            self.LimitOrder(symbol, quantity, take_profit_price)
    
    def update_metrics(self, trade_result: Dict) -> None:
        """Update trading metrics after a trade is closed.
        
        Args:
            trade_result: Dictionary containing trade results
        """
        self.metrics['total_trades'] += 1
        
        if trade_result['profit'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['avg_win'] = ((self.metrics['avg_win'] * (self.metrics['winning_trades'] - 1)) + 
                                     trade_result['profit']) / self.metrics['winning_trades']
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['avg_loss'] = ((self.metrics['avg_loss'] * (self.metrics['losing_trades'] - 1)) + 
                                      trade_result['profit']) / self.metrics['losing_trades']
        
        # Update max drawdown
        current_drawdown = self.Portfolio.TotalPortfolioValue / self.Portfolio.MaximumPortfolioValue - 1
        self.metrics['max_drawdown'] = min(self.metrics['max_drawdown'], current_drawdown)
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached.
        
        Returns:
            bool: True if within risk limits, False otherwise
        """
        # Check number of positions
        if len([x for x in self.Portfolio.Values if x.Invested]) >= self.max_positions:
            return False
            
        # Check drawdown
        if abs(self.metrics['max_drawdown']) > self.max_drawdown:
            return False
            
        return True 
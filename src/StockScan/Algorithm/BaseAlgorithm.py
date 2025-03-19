"""
BaseAlgorithm.py - Base Algorithm Implementation

This module provides the base implementation for all LEAN-based trading algorithms.
It implements the IAlgorithm interface and provides common functionality used by all algorithms.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Import LEAN components using pythonnet
try:
    import clr
    # Add references to LEAN assemblies
    clr.AddReference("QuantConnect.Common")
    clr.AddReference("QuantConnect.Algorithm")
    clr.AddReference("QuantConnect.Indicators")
    clr.AddReference("QuantConnect.Algorithm.Framework")
    
    from QuantConnect import *
    from QuantConnect.Algorithm import QCAlgorithm
    from QuantConnect.Data import *
    from QuantConnect.Data.Market import *
    from QuantConnect.Parameters import *
    from QuantConnect.Securities import *
    from QuantConnect.Indicators import *
    
    LEAN_AVAILABLE = True
except ImportError:
    logging.warning("QuantConnect LEAN framework import failed: No module named 'clr'")
    logging.warning("Some features may not be available")
    LEAN_AVAILABLE = False

from src.StockScan.Interfaces.IAlgorithm import IAlgorithm

# Set up logger
logger = logging.getLogger(__name__)


class BaseAlgorithm(QCAlgorithm, IAlgorithm):
    """
    Base class for LEAN-based trading algorithms.
    Implements the IAlgorithm interface and inherits from QCAlgorithm.
    
    This class provides common functionality used by all algorithms in StockScan,
    following LEAN best practices.
    """
    
    def __init__(self):
        """Initialize the algorithm."""
        super().__init__()
        
        # Trading parameters with default values
        self.position_size = 0.1  # Default position size as fraction of portfolio
        self.stop_loss_pct = 0.02  # Default stop loss percentage
        self.take_profit_pct = 0.05  # Default take profit percentage
        
        # Risk management settings
        self.max_positions = 5  # Default maximum number of positions
        self.max_drawdown = 0.15  # Default maximum drawdown
        
        # Performance tracking
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0
        }
        
        # Indicators dictionary
        self.indicators = {}
        
        # Initialize symbol list
        self.symbols = []
    
    def Initialize(self) -> None:
        """
        Initialize the algorithm settings and state.
        Must be overridden by derived classes.
        """
        # Set default dates
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(datetime.now().year, datetime.now().month, datetime.now().day)
        
        # Set default cash
        self.SetCash(100000)
        
        # Set brokerage model
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        
        # Add a default universe
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
    
    def CoarseSelectionFunction(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        """
        Coarse universe selection function.
        
        Args:
            coarse: List of coarse fundamental data
            
        Returns:
            List of symbols selected for the fine universe
        """
        # Default implementation just returns top traded stocks by dollar volume
        sorted_by_dollar_volume = sorted(coarse, 
                                         key=lambda x: x.DollarVolume, 
                                         reverse=True)
        
        # Limit to top N
        return [x.Symbol for x in sorted_by_dollar_volume[:20]]
    
    def FineSelectionFunction(self, fine: List[FineFundamental]) -> List[Symbol]:
        """
        Fine universe selection function.
        
        Args:
            fine: List of fine fundamental data
            
        Returns:
            List of symbols selected for the algorithm universe
        """
        # Default implementation just returns what was passed in
        return [x.Symbol for x in fine]
    
    def OnData(self, data: Slice) -> None:
        """
        Event handler for market data updates.
        Must be overridden by derived classes to implement trading logic.
        
        Args:
            data: The new data slice
        """
        pass
    
    def OnOrderEvent(self, order_event: OrderEvent) -> None:
        """
        Event handler for order events (fills, cancellations, etc.)
        
        Args:
            order_event: The order event information
        """
        # Log the order event
        self.Log(f"Order event: {order_event.Symbol}: {order_event.Status}")
        
        # Update metrics for filled orders
        if order_event.Status == OrderStatus.Filled:
            self.metrics['total_trades'] += 1
            
            # Get order profit/loss for closed trades
            order = self.Transactions.GetOrderById(order_event.OrderId)
            if order and order.Type == OrderType.Market:
                symbol = order_event.Symbol
                if not self.Portfolio[symbol].Invested:
                    # Position was closed
                    trade_pl = self.GetTradeProfit(symbol)
                    if trade_pl > 0:
                        self.metrics['winning_trades'] += 1
                        self.metrics['avg_win'] = ((self.metrics['avg_win'] * (self.metrics['winning_trades'] - 1)) + 
                                                 trade_pl) / self.metrics['winning_trades']
                    else:
                        self.metrics['losing_trades'] += 1
                        self.metrics['avg_loss'] = ((self.metrics['avg_loss'] * (self.metrics['losing_trades'] - 1)) + 
                                                  trade_pl) / self.metrics['losing_trades']
    
    def OnEndOfDay(self, symbol: Symbol = None) -> None:
        """
        Event handler called at the end of each trading day.
        
        Args:
            symbol: The symbol for which the day has ended
        """
        # Track portfolio performance
        self.metrics['max_drawdown'] = min(
            self.metrics['max_drawdown'],
            (self.Portfolio.TotalPortfolioValue / self.Portfolio.MaximumPortfolioValue) - 1
        )
        
        # Log daily performance
        self.Log(f"End of day - Portfolio value: ${self.Portfolio.TotalPortfolioValue}")
    
    def OnEndOfAlgorithm(self) -> None:
        """
        Event handler called at the end of the algorithm execution.
        Use this for final reporting.
        """
        # Log overall performance
        self.Log("Algorithm completed")
        self.Log(f"Final portfolio value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"Total trades: {self.metrics['total_trades']}")
        self.Log(f"Win rate: {self.metrics['winning_trades'] / max(1, self.metrics['total_trades']):.2%}")
        self.Log(f"Maximum drawdown: {self.metrics['max_drawdown']:.2%}")
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the algorithm's universe.
        
        Args:
            symbol: Symbol to add
        """
        equity = self.AddEquity(symbol, Resolution.Daily)
        self.symbols.append(equity.Symbol)
    
    def calculate_position_size(self, symbol: Symbol) -> float:
        """
        Calculate position size based on portfolio value and risk parameters.
        
        Args:
            symbol: Symbol to calculate position size for
            
        Returns:
            Position size in number of shares
        """
        portfolio_value = self.Portfolio.TotalPortfolioValue
        price = self.Securities[symbol].Price
        
        # Calculate position size based on portfolio percentage
        position_value = portfolio_value * self.position_size
        return int(position_value / price)
    
    def place_trade(self, symbol: Symbol, direction: str, quantity: float) -> None:
        """
        Place a trade with stop loss and take profit orders.
        
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
            stop_price = current_price * (1 - self.stop_loss_pct)
            take_profit_price = current_price * (1 + self.take_profit_pct)
            
            # Main order
            order_ticket = self.MarketOrder(symbol, quantity)
            if order_ticket.Status == OrderStatus.Filled or order_ticket.Status == OrderStatus.PartiallyFilled:
                # Add stop loss
                self.StopMarketOrder(symbol, -quantity, stop_price)
                # Add take profit
                self.LimitOrder(symbol, -quantity, take_profit_price)
            
        else:  # short
            # Place short order with stop loss and take profit
            stop_price = current_price * (1 + self.stop_loss_pct)
            take_profit_price = current_price * (1 - self.take_profit_pct)
            
            # Main order
            order_ticket = self.MarketOrder(symbol, -quantity)
            if order_ticket.Status == OrderStatus.Filled or order_ticket.Status == OrderStatus.PartiallyFilled:
                # Add stop loss
                self.StopMarketOrder(symbol, quantity, stop_price)
                # Add take profit
                self.LimitOrder(symbol, quantity, take_profit_price)
    
    def check_risk_limits(self) -> bool:
        """
        Check if risk limits are breached.
        
        Returns:
            bool: True if within risk limits, False otherwise
        """
        # Check number of positions
        if len([x for x in self.Portfolio.Values if x.Invested]) >= self.max_positions:
            return False
            
        # Check drawdown
        current_drawdown = self.Portfolio.TotalPortfolioValue / self.Portfolio.MaximumPortfolioValue - 1
        if abs(current_drawdown) > self.max_drawdown:
            return False
            
        return True
    
    def GetTradeProfit(self, symbol: Symbol) -> float:
        """
        Calculate profit/loss for a closed trade.
        
        Args:
            symbol: Symbol to calculate profit for
            
        Returns:
            Profit/loss amount
        """
        trades = self.Transactions.GetOrdersForSymbol(symbol)
        if not trades:
            return 0
            
        # Calculate P&L
        buy_value = 0
        sell_value = 0
        buy_quantity = 0
        sell_quantity = 0
        
        for order in trades:
            if order.Direction == OrderDirection.Buy:
                buy_value += order.Price * order.Quantity
                buy_quantity += order.Quantity
            else:
                sell_value += order.Price * order.Quantity
                sell_quantity += order.Quantity
        
        # Return profit
        return sell_value - buy_value
    
    # Implementation of remaining IAlgorithm methods (these directly use QCAlgorithm methods)
    def SetStartDate(self, year: int, month: int, day: int) -> None:
        super().SetStartDate(year, month, day)
    
    def SetEndDate(self, year: int, month: int, day: int) -> None:
        super().SetEndDate(year, month, day)
    
    def SetCash(self, amount: float) -> None:
        super().SetCash(amount)
    
    def AddEquity(self, symbol: str, resolution: Any) -> Any:
        return super().AddEquity(symbol, resolution)
    
    def SetBrokerageModel(self, brokerage_model: Any) -> None:
        super().SetBrokerageModel(brokerage_model) 
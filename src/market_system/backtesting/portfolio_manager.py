"""
Portfolio manager for backtesting.
Handles portfolio state, positions, and capital allocation.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import uuid
from dataclasses import dataclass, field, asdict
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Position data class for tracking trades."""
    id: str
    symbol: str
    action: str  # "BUY" or "SELL"
    entry_price: Decimal
    quantity: Decimal
    entry_timestamp: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    exit_timestamp: Optional[datetime] = None
    strategy_name: str = ""
    pnl: Optional[Decimal] = None
    pnl_percentage: Optional[Decimal] = None
    status: str = "OPEN"  # "OPEN", "CLOSED"
    metadata: Dict = field(default_factory=dict)
    
    def calculate_pnl(self, current_price: Decimal) -> Tuple[Decimal, Decimal]:
        """Calculate P&L for the position."""
        if self.action == "BUY":
            pnl = (current_price - self.entry_price) * self.quantity
            pnl_percentage = (current_price / self.entry_price - Decimal("1.0")) * Decimal("100.0")
        else:  # SELL
            pnl = (self.entry_price - current_price) * self.quantity
            pnl_percentage = (Decimal("1.0") - current_price / self.entry_price) * Decimal("100.0")
            
        return pnl, pnl_percentage
        
    def close(self, exit_price: Decimal, exit_timestamp: datetime) -> None:
        """Close the position."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.status = "CLOSED"
        
        # Calculate final P&L
        self.pnl, self.pnl_percentage = self.calculate_pnl(exit_price)
        
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return asdict(self)

class PortfolioManager:
    """
    Manages portfolio state, positions, and capital allocation.
    Tracks open positions, executes trades, and calculates portfolio metrics.
    """
    
    def __init__(self, initial_capital: Decimal):
        """
        Initialize the portfolio manager.
        
        Args:
            initial_capital: Initial capital for the portfolio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Portfolio metrics
        self.max_drawdown = Decimal("0.0")
        self.peak_equity = initial_capital
        self.current_drawdown = Decimal("0.0")
        
        logger.info(f"PortfolioManager initialized with capital: {initial_capital}")
        
    def open_position(self, 
                      symbol: str, 
                      action: str, 
                      price: Decimal, 
                      quantity: Decimal,
                      timestamp: datetime,
                      stop_loss: Optional[Decimal] = None,
                      take_profit: Optional[Decimal] = None,
                      strategy_name: str = "",
                      metadata: Optional[Dict] = None) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            price: Entry price
            quantity: Position size
            timestamp: Entry timestamp
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            strategy_name: Name of the strategy opening the position
            metadata: Additional metadata for the position
            
        Returns:
            The newly created Position
        """
        # Generate a unique ID for the position
        position_id = str(uuid.uuid4())
        
        # Create the position
        position = Position(
            id=position_id,
            symbol=symbol,
            action=action,
            entry_price=price,
            quantity=quantity,
            entry_timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            metadata=metadata or {}
        )
        
        # Calculate position cost
        position_cost = price * quantity
        
        # Update capital
        if action == "BUY":
            self.current_capital -= position_cost
        else:  # SELL (for short positions)
            # For simplicity, we're not handling margin requirements for shorts
            # In a real system, you'd reserve margin and handle it properly
            self.current_capital += position_cost
        
        # Store the position
        self.positions[position_id] = position
        
        # Log the trade
        trade_record = {
            "type": "OPEN",
            "position_id": position_id,
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "timestamp": timestamp,
            "strategy": strategy_name
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Opened {action} position for {symbol}: {quantity} @ {price} (ID: {position_id})")
        
        return position
        
    def close_position(self, 
                       position_id: str, 
                       price: Decimal,
                       timestamp: datetime) -> Optional[Position]:
        """
        Close an existing position.
        
        Args:
            position_id: ID of the position to close
            price: Exit price
            timestamp: Exit timestamp
            
        Returns:
            The closed Position or None if position not found
        """
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return None
            
        position = self.positions[position_id]
        
        # Close the position
        position.close(price, timestamp)
        
        # Update capital
        if position.action == "BUY":
            self.current_capital += price * position.quantity
        else:  # SELL (short position)
            self.current_capital -= price * position.quantity
            
        # Move to closed positions
        del self.positions[position_id]
        self.closed_positions.append(position)
        
        # Log the trade
        trade_record = {
            "type": "CLOSE",
            "position_id": position_id,
            "symbol": position.symbol,
            "action": "SELL" if position.action == "BUY" else "BUY",  # Opposite of entry action
            "price": price,
            "quantity": position.quantity,
            "timestamp": timestamp,
            "pnl": position.pnl,
            "pnl_percentage": position.pnl_percentage
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Closed position {position_id}: {position.symbol} @ {price} (P&L: {position.pnl})")
        
        return position
        
    def update_positions(self, 
                         timestamp: datetime, 
                         market_data: Dict[str, Union[Dict, pd.DataFrame]]) -> List[Position]:
        """
        Update all positions with the latest market data.
        Check for stop loss/take profit triggers.
        
        Args:
            timestamp: Current timestamp
            market_data: Dict mapping symbols to market data (either Dict or DataFrame)
            
        Returns:
            List of positions that were closed
        """
        closed_positions = []
        
        for position_id, position in list(self.positions.items()):
            symbol = position.symbol
            
            # Skip if no market data for this symbol
            if symbol not in market_data:
                continue
                
            # Get current price
            if isinstance(market_data[symbol], pd.DataFrame):
                # If market_data is a DataFrame, get the last row's close price
                current_price = Decimal(str(market_data[symbol].iloc[-1]['close']))
            else:
                # If market_data is a Dict, get the close price directly
                current_price = Decimal(str(market_data[symbol]["close"]))
            
            # Check stop loss
            if position.stop_loss is not None:
                if (position.action == "BUY" and current_price <= position.stop_loss) or \
                   (position.action == "SELL" and current_price >= position.stop_loss):
                    # Stop loss triggered
                    closed_position = self.close_position(position_id, position.stop_loss, timestamp)
                    if closed_position:
                        closed_positions.append(closed_position)
                    continue
                    
            # Check take profit
            if position.take_profit is not None:
                if (position.action == "BUY" and current_price >= position.take_profit) or \
                   (position.action == "SELL" and current_price <= position.take_profit):
                    # Take profit triggered
                    closed_position = self.close_position(position_id, position.take_profit, timestamp)
                    if closed_position:
                        closed_positions.append(closed_position)
                    continue
        
        return closed_positions
        
    def calculate_equity(self, 
                         timestamp: datetime, 
                         market_data: Dict[str, Union[Dict, pd.DataFrame]]) -> Decimal:
        """
        Calculate current equity based on cash and positions.
        
        Args:
            timestamp: Current timestamp
            market_data: Dict mapping symbols to market data (either Dict or DataFrame)
            
        Returns:
            Current equity
        """
        # Start with cash
        equity = self.current_capital
        
        # Add value of open positions
        for position in self.positions.values():
            symbol = position.symbol
            
            # Skip if no market data for this symbol
            if symbol not in market_data:
                continue
                
            # Get current price
            if isinstance(market_data[symbol], pd.DataFrame):
                # If market_data is a DataFrame, get the last row's close price
                current_price = Decimal(str(market_data[symbol].iloc[-1]['close']))
            else:
                # If market_data is a Dict, get the close price directly
                current_price = Decimal(str(market_data[symbol]["close"]))
            
            # Calculate position value
            if position.action == "BUY":
                position_value = position.quantity * current_price
            else:  # SELL
                # For short positions, value increases as price decreases
                position_value = position.quantity * (position.entry_price * Decimal("2.0") - current_price)
                
            equity += position_value
            
        # Update drawdown metrics
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = Decimal("0.0")
        else:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                
        # Record equity history
        equity_record = {
            "timestamp": timestamp,
            "equity": equity,
            "drawdown": self.current_drawdown
        }
        self.equity_history.append(equity_record)
        
        return equity
        
    def get_position_sizing(self, 
                            symbol: str, 
                            price: Decimal, 
                            risk_percentage: Union[Decimal, float],
                            stop_loss: Optional[Decimal] = None) -> Decimal:
        """
        Calculate position size based on risk percentage.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            risk_percentage: Percentage of capital to risk
            stop_loss: Stop loss price (for risk calculation)
            
        Returns:
            Position size (quantity)
        """
        # Ensure risk_percentage is a Decimal
        if not isinstance(risk_percentage, Decimal):
            risk_percentage = Decimal(str(risk_percentage))
            
        # Make position sizing more aggressive by multiplying risk percentage
        risk_percentage = risk_percentage * Decimal("5.0")
            
        # Calculate risk amount
        risk_amount = self.current_capital * risk_percentage / Decimal("100.0")
        
        # If stop loss is provided, calculate position size based on risk per trade
        if stop_loss is not None:
            risk_per_unit = abs(price - stop_loss)
            if risk_per_unit > Decimal("0.0"):
                quantity = risk_amount / risk_per_unit
                return quantity
                
        # Default to percentage of capital
        quantity = risk_amount / price
        return quantity
        
    def get_portfolio_stats(self) -> Dict:
        """
        Get portfolio statistics.
        
        Returns:
            Dict with portfolio statistics
        """
        stats = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "current_equity": self.equity_history[-1]["equity"] if self.equity_history else self.current_capital,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "total_trades": len(self.trade_history),
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity
        }
        
        # Calculate win/loss stats
        if self.closed_positions:
            winning_trades = [p for p in self.closed_positions if p.pnl > Decimal("0.0")]
            losing_trades = [p for p in self.closed_positions if p.pnl <= Decimal("0.0")]
            
            stats["winning_trades"] = len(winning_trades)
            stats["losing_trades"] = len(losing_trades)
            stats["win_rate"] = Decimal(str(len(winning_trades) / len(self.closed_positions)))
            
            if winning_trades:
                stats["avg_win"] = sum(p.pnl for p in winning_trades) / len(winning_trades)
            else:
                stats["avg_win"] = Decimal("0.0")
                
            if losing_trades:
                stats["avg_loss"] = sum(p.pnl for p in losing_trades) / len(losing_trades)
            else:
                stats["avg_loss"] = Decimal("0.0")
                
            if stats["avg_loss"] != Decimal("0.0"):
                stats["profit_factor"] = abs(stats["avg_win"] * stats["winning_trades"] / (stats["avg_loss"] * stats["losing_trades"]))
            else:
                stats["profit_factor"] = Decimal("0.0")
                
        return stats 

    def update_portfolio(self, execution: Dict) -> Optional[Position]:
        """
        Update portfolio based on execution results.
        
        Args:
            execution: Execution details from ExecutionHandler
            
        Returns:
            The opened position or None if failed
        """
        symbol = execution["symbol"]
        action = execution["action"]
        price = execution["executed_price"]
        quantity = execution["quantity"]
        timestamp = execution["timestamp"]
        stop_loss = execution.get("stop_loss")
        take_profit = execution.get("take_profit")
        
        # Calculate cost
        cost = price * quantity
        
        # Check if we have enough capital
        if action == "BUY" and cost > self.current_capital:
            logger.warning(f"Not enough capital to open position: {symbol} {action} {quantity} @ {price}")
            return None
            
        # Open position
        position = self.open_position(
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name="",  # Strategy name not available in execution
            metadata={"execution": execution}
        )
        
        return position 
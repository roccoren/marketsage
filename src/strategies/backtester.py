import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base import Strategy

class Trade:
    def __init__(self, entry_price: float, entry_date: pd.Timestamp, 
                 signal_type: str, position_size: float = 1.0):
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.exit_price: Optional[float] = None
        self.exit_date: Optional[pd.Timestamp] = None
        self.signal_type = signal_type
        self.position_size = position_size
        self.pnl: Optional[float] = None
        self.return_pct: Optional[float] = None

    def close_trade(self, exit_price: float, exit_date: pd.Timestamp):
        self.exit_price = exit_price
        self.exit_date = exit_date
        
        # Calculate P&L and return
        multiplier = 1 if self.signal_type == "BUY" else -1
        self.pnl = (exit_price - self.entry_price) * self.position_size * multiplier
        self.return_pct = (exit_price / self.entry_price - 1) * 100 * multiplier

class BacktestResult:
    def __init__(self):
        self.trades: List[Trade] = []
        self.metrics: Dict = {}
    
    def calculate_metrics(self):
        if not self.trades:
            return
            
        # Extract returns and P&Ls
        returns = [trade.return_pct for trade in self.trades if trade.return_pct is not None]
        pnls = [trade.pnl for trade in self.trades if trade.pnl is not None]
        
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([pnl for pnl in pnls if pnl > 0])
        self.metrics['losing_trades'] = len([pnl for pnl in pnls if pnl < 0])
        
        # Win rate and profit metrics
        self.metrics['win_rate'] = self.metrics['winning_trades'] / len(self.trades)
        self.metrics['total_pnl'] = sum(pnls)
        self.metrics['avg_return'] = np.mean(returns)
        self.metrics['avg_winning_return'] = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        self.metrics['avg_losing_return'] = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        
        # Risk metrics
        self.metrics['max_drawdown'] = self._calculate_max_drawdown(pnls)
        self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        self.metrics['profit_factor'] = self._calculate_profit_factor(pnls)
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        cumulative = np.cumsum(pnls)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = rolling_max - cumulative
        return max(drawdowns) if len(drawdowns) > 0 else 0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        if not returns:
            return 0
        excess_returns = np.array(returns) - (risk_free_rate / 252)  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        gross_profits = sum(pnl for pnl in pnls if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in pnls if pnl < 0))
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')

class Backtester:
    def __init__(self, strategy: Strategy, initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
    
    def run(self, data: pd.DataFrame, position_size: float = 0.1) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            position_size: Fraction of capital to use per trade (0.1 = 10%)
            
        Returns:
            BacktestResult object with trades and metrics
        """
        # Reset state
        self.current_capital = self.initial_capital
        self.open_trades = []
        self.closed_trades = []
        
        # Run strategy day by day
        for i in range(len(data)):
            current_slice = data.iloc[:i+1]
            date = current_slice.index[-1]
            
            # Check for exit signals on open trades
            self._process_exits(current_slice, date)
            
            # Look for new signals
            signals = self.strategy.analyze(current_slice)
            
            for signal in signals:
                # Calculate position size based on current capital
                trade_value = self.current_capital * position_size
                position_size_units = trade_value / signal['Price']
                
                # Open new trade
                trade = Trade(
                    entry_price=signal['Price'],
                    entry_date=date,
                    signal_type=signal['Type'],
                    position_size=position_size_units
                )
                self.open_trades.append(trade)
        
        # Close any remaining open trades at the last price
        if self.open_trades:
            last_price = data['Close'].iloc[-1]
            last_date = data.index[-1]
            for trade in self.open_trades:
                trade.close_trade(last_price, last_date)
                self.closed_trades.append(trade)
            self.open_trades = []
        
        # Calculate backtest results
        result = BacktestResult()
        result.trades = self.closed_trades
        result.calculate_metrics()
        
        return result
    
    def _process_exits(self, data: pd.DataFrame, current_date: pd.Timestamp):
        """Process exit signals for open trades."""
        if not self.open_trades:
            return
            
        current_price = data['Close'].iloc[-1]
        signals = self.strategy.analyze(data)
        
        # Check if we have opposing signals to our open positions
        remaining_trades = []
        for trade in self.open_trades:
            should_exit = False
            
            for signal in signals:
                if ((trade.signal_type == "BUY" and signal['Type'] == "SELL") or
                    (trade.signal_type == "SELL" and signal['Type'] == "BUY")):
                    should_exit = True
                    break
            
            if should_exit:
                trade.close_trade(current_price, current_date)
                self.closed_trades.append(trade)
                
                # Update capital
                self.current_capital += trade.pnl
            else:
                remaining_trades.append(trade)
        
        self.open_trades = remaining_trades
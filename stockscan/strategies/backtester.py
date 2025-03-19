import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from .base import Strategy

# Configure module logger
logger = logging.getLogger(__name__)

class Trade:
    def __init__(self, entry_price: float, entry_date: pd.Timestamp, 
                 exit_price: float = None, exit_date: pd.Timestamp = None):
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.return_pct = None
        self.pnl = None
        
    def close(self, exit_price: float, exit_date: pd.Timestamp):
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.return_pct = (self.exit_price - self.entry_price) / self.entry_price
        self.pnl = self.exit_price - self.entry_price

class BacktestResult:
    """
    Class to store backtest results and provide metrics
    """
    def __init__(self, 
                 equity_curve: pd.Series, 
                 trades: List[Dict[str, Any]], 
                 positions: pd.DataFrame, 
                 strategy_name: str,
                 symbol: str = None,
                 initial_capital: float = 100000.0):
        self.equity_curve = equity_curve
        self.trades = trades
        self.positions = positions
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.metrics = self._calculate_metrics()
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results"""
        metrics = {}
        
        # Skip if no trades
        if not self.trades:
            logger.warning("No trades to calculate metrics")
            return {
                'total_trades': 0,
                'net_profit_pct': 0,
                'win_rate': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        metrics['total_trades'] = len(self.trades)
        metrics['total_bars'] = len(self.equity_curve)
        
        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()
        
        # Calculate profit metrics
        final_equity = self.equity_curve.iloc[-1]
        metrics['net_profit'] = final_equity - self.initial_capital
        metrics['net_profit_pct'] = (metrics['net_profit'] / self.initial_capital) * 100
        metrics['annualized_return_pct'] = self._calculate_annualized_return(self.equity_curve) * 100
        
        # Risk metrics
        metrics['max_drawdown_pct'] = self._calculate_max_drawdown(self.equity_curve) * 100
        
        # Win/loss metrics
        winning_trades = [t for t in self.trades if t.get('PnL', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('PnL', 0) <= 0]
        
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        else:
            metrics['win_rate'] = 0
            
        # Calculate profit metrics for winning and losing trades
        if winning_trades:
            metrics['avg_win'] = sum(t.get('PnL', 0) for t in winning_trades) / len(winning_trades)
            metrics['max_win'] = max(t.get('PnL', 0) for t in winning_trades)
        else:
            metrics['avg_win'] = 0
            metrics['max_win'] = 0
            
        if losing_trades:
            metrics['avg_loss'] = sum(t.get('PnL', 0) for t in losing_trades) / len(losing_trades)
            metrics['max_loss'] = min(t.get('PnL', 0) for t in losing_trades)
        else:
            metrics['avg_loss'] = 0
            metrics['max_loss'] = 0
        
        # Calculate profit factor
        total_gross_profit = sum(t.get('PnL', 0) for t in winning_trades)
        total_gross_loss = abs(sum(t.get('PnL', 0) for t in losing_trades))
        
        if total_gross_loss > 0:
            metrics['profit_factor'] = total_gross_profit / total_gross_loss
        else:
            metrics['profit_factor'] = float('inf') if total_gross_profit > 0 else 0
        
        # Calculate risk-adjusted metrics
        if len(returns) > 1:
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
            metrics['volatility'] = 0
        
        # Average trade metrics
        if metrics['total_trades'] > 0:
            avg_trade_pnl = sum(t.get('PnL', 0) for t in self.trades) / metrics['total_trades']
            metrics['avg_trade_pnl'] = avg_trade_pnl
            metrics['avg_trade_pnl_pct'] = (avg_trade_pnl / self.initial_capital) * 100
            
        # Add time-related metrics
        if self.trades and len(self.trades) > 1:
            start_date = min(t.get('EntryDate', datetime.now()) for t in self.trades)
            end_date = max(t.get('ExitDate', datetime.now()) for t in self.trades if 'ExitDate' in t)
            metrics['backtest_length_days'] = (end_date - start_date).days
        
        # Calculate average holding period
        if self.trades:
            holding_periods = []
            for trade in self.trades:
                if 'EntryDate' in trade and 'ExitDate' in trade:
                    holding_period = (trade['ExitDate'] - trade['EntryDate']).days
                    holding_periods.append(holding_period)
            
            if holding_periods:
                metrics['avg_holding_period_days'] = sum(holding_periods) / len(holding_periods)
                metrics['max_holding_period_days'] = max(holding_periods)
                metrics['min_holding_period_days'] = min(holding_periods)
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        # Running maximum
        running_max = equity_curve.expanding().max()
        # Drawdown in percentage terms
        drawdown = (equity_curve / running_max - 1)
        # Maximum drawdown
        max_drawdown = drawdown.min()
        return abs(max_drawdown) if not np.isnan(max_drawdown) else 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (using only downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino
    
    def _calculate_annualized_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return"""
        if len(equity_curve) <= 1:
            return 0
            
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        
        # Calculate years
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        days = (end_date - start_date).days
        
        if days <= 0:
            return 0
            
        years = days / 365.25
        
        # Annualize
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
            return annualized_return
        else:
            return 0

class Backtester:
    """
    Backtester class that simulates trading with signals from a strategy
    """
    def __init__(self, 
                 strategy: Strategy, 
                 initial_capital: float = 100000, 
                 position_size: float = 1.0,
                 commission: float = 0.0,
                 slippage: float = 0.0,
                 enable_stop_loss: bool = True,
                 enable_take_profit: bool = True):
        """
        Initialize the backtester
        
        Args:
            strategy: Strategy object that generates signals
            initial_capital: Initial capital amount
            position_size: Position size as decimal (0.0-1.0), or fixed position size if > 1.0
            commission: Commission per trade (% as decimal)
            slippage: Slippage per trade (% as decimal)
            enable_stop_loss: Whether to use stop loss levels
            enable_take_profit: Whether to use take profit levels
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.enable_stop_loss = enable_stop_loss
        self.enable_take_profit = enable_take_profit
        self.current_position = 0
        self.entry_price = 0
        self.entry_date = None
        self.stop_loss = None
        self.take_profit = None
        self.equity = self.initial_capital
        self.equity_curve = None
        self.trades = []
        
    def run(self, 
            data: pd.DataFrame, 
            symbol: str = None,
            start_date: Optional[pd.Timestamp] = None,
            end_date: Optional[pd.Timestamp] = None) -> BacktestResult:
        """
        Run backtest on given data
        
        Args:
            data: DataFrame containing OHLCV data
            symbol: Optional symbol name
            start_date: Optional start date for backtesting
            end_date: Optional end date for backtesting
            
        Returns:
            BacktestResult object containing backtest results
        """
        # Prepare the data
        df = self.strategy.prepare_data(data)
        
        # Filter date range if specified
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
            
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        if df.empty:
            logger.warning("No data available for backtesting after filtering")
            return None
        
        # Get strategy signals
        signals = self.strategy.analyze(df)
        
        # Initialize result tracking
        equity_history = pd.Series(self.initial_capital, index=df.index)
        positions = pd.DataFrame(0, index=df.index, columns=['Position', 'EntryPrice', 'StopLoss', 'TakeProfit'])
        self.equity = self.initial_capital
        self.current_position = 0
        self.trades = []
        
        # Process each bar
        for i, (date, row) in enumerate(df.iterrows()):
            # Process any pending stops before applying new signals
            if self.current_position != 0:
                # Check for stop loss hit
                if self.enable_stop_loss and self.stop_loss is not None:
                    if (self.current_position > 0 and row['Low'] <= self.stop_loss) or \
                       (self.current_position < 0 and row['High'] >= self.stop_loss):
                        # Stop loss hit
                        exit_price = self.stop_loss
                        self._exit_position(date, exit_price, "StopLoss")
                
                # Check for take profit hit
                elif self.enable_take_profit and self.take_profit is not None:
                    if (self.current_position > 0 and row['High'] >= self.take_profit) or \
                       (self.current_position < 0 and row['Low'] <= self.take_profit):
                        # Take profit hit
                        exit_price = self.take_profit
                        self._exit_position(date, exit_price, "TakeProfit")
            
            # Find any signal for current date
            current_signals = [s for s in signals if s['Date'] == date]
            
            # Apply signals if any
            for signal in current_signals:
                signal_type = signal['Type']
                
                # Exit current position if needed
                if (signal_type == 'SELL' and self.current_position > 0) or \
                   (signal_type == 'BUY' and self.current_position < 0):
                    self._exit_position(date, row['Close'], "Signal")
                
                # Enter new position if not already in that direction
                if (signal_type == 'BUY' and self.current_position <= 0) or \
                   (signal_type == 'SELL' and self.current_position >= 0):
                    self._enter_position(date, row['Close'], signal)
            
            # Update equity curve based on current price
            if self.current_position != 0:
                unrealized_pnl = self._calculate_unrealized_pnl(row['Close'])
                current_equity = self.equity + unrealized_pnl
            else:
                current_equity = self.equity
                
            equity_history[date] = current_equity
            
            # Update positions dataframe
            positions.loc[date, 'Position'] = self.current_position
            if self.current_position != 0:
                positions.loc[date, 'EntryPrice'] = self.entry_price
                positions.loc[date, 'StopLoss'] = self.stop_loss
                positions.loc[date, 'TakeProfit'] = self.take_profit
            
        # Close any open position at the end
        if self.current_position != 0:
            last_date = df.index[-1]
            last_price = df.iloc[-1]['Close']
            self._exit_position(last_date, last_price, "EndOfData")
        
        # Create backtest result
        result = BacktestResult(
            equity_curve=equity_history, 
            trades=self.trades, 
            positions=positions,
            strategy_name=self.strategy.name,
            symbol=symbol,
            initial_capital=self.initial_capital
        )
        
        return result
    
    def _enter_position(self, date: pd.Timestamp, price: float, signal: Dict[str, Any]) -> None:
        """Enter a new position"""
        # Calculate position size
        if self.position_size <= 1.0:
            # Percentage of equity
            amount = self.equity * self.position_size
        else:
            # Fixed size
            amount = self.position_size
            
        # Adjust entry price for slippage
        if signal['Type'] == 'BUY':
            entry_price = price * (1 + self.slippage)
            self.current_position = amount / entry_price
        else:  # SELL
            entry_price = price * (1 - self.slippage)
            self.current_position = -amount / entry_price

        # Apply commission
        self.equity -= abs(amount) * self.commission
            
        # Set entry details
        self.entry_price = entry_price
        self.entry_date = date
        
        # Set stop loss and take profit if provided
        self.stop_loss = signal.get('StopLoss')
        self.take_profit = signal.get('TakeProfit')
        
        # Create trade entry in log
        self.trades.append({
            'EntryDate': date,
            'EntryPrice': entry_price,
            'Direction': signal['Type'],
            'Size': abs(self.current_position),
            'StopLoss': self.stop_loss,
            'TakeProfit': self.take_profit
        })
        
    def _exit_position(self, date: pd.Timestamp, price: float, reason: str) -> None:
        """Exit current position"""
        if self.current_position == 0:
            return
            
        # Adjust exit price for slippage
        if self.current_position > 0:
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
            
        # Calculate PnL
        pnl = self._calculate_pnl(exit_price)
        
        # Apply commission
        commission_amount = abs(self.current_position * exit_price) * self.commission
        pnl -= commission_amount
        
        # Update equity
        self.equity += pnl
        
        # Update trade log
        if self.trades:
            self.trades[-1].update({
                'ExitDate': date,
                'ExitPrice': exit_price,
                'ExitReason': reason,
                'PnL': pnl,
                'PnL_Pct': (pnl / (abs(self.current_position) * self.entry_price)) * 100,
                'Commission': commission_amount
            })
        
        # Reset position
        self.current_position = 0
        self.entry_price = 0
        self.entry_date = None
        self.stop_loss = None
        self.take_profit = None
        
    def _calculate_pnl(self, exit_price: float) -> float:
        """Calculate PnL for current position"""
        if self.current_position > 0:
            pnl = self.current_position * (exit_price - self.entry_price)
        else:
            pnl = self.current_position * (self.entry_price - exit_price)
            
        return pnl
        
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for current position"""
        return self._calculate_pnl(current_price)
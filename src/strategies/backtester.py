import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base import Strategy
import logging

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
    def __init__(self):
        self.trades: List[Trade] = []
        self.metrics: Dict = {}
        
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        
    def calculate_metrics(self):
        """Calculate performance metrics for the backtest"""
        logger.info("Calculating backtest metrics...")
        
        # Initialize default metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'recovery_period': 0,
            'avg_winning_return': 0.0,
            'avg_losing_return': 0.0
        }
        
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return
            
        # Extract returns and P&Ls
        returns = [trade.return_pct for trade in self.trades if trade.return_pct is not None]
        pnls = [trade.pnl for trade in self.trades if trade.pnl is not None]
        
        logger.debug(f"Number of trades with returns: {len(returns)}")
        logger.debug(f"Number of trades with P&L: {len(pnls)}")
        
        if not returns or not pnls:
            logger.warning("No completed trades found for metric calculation")
            return
        
        # Basic metrics
        logger.debug("Calculating basic metrics...")
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([pnl for pnl in pnls if pnl > 0])
        self.metrics['losing_trades'] = len([pnl for pnl in pnls if pnl < 0])
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        self.metrics['total_pnl'] = sum(pnls)
        self.metrics['avg_return'] = np.mean(returns)
        self.metrics['avg_winning_return'] = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        self.metrics['avg_losing_return'] = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        
        logger.debug(f"Win rate: {self.metrics['win_rate']:.2%}")
        logger.debug(f"Total P&L: {self.metrics['total_pnl']:.2f}")
        
        # Risk metrics
        logger.debug("Calculating risk metrics...")
        self.metrics['max_drawdown'] = self._calculate_max_drawdown(pnls)
        self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        self.metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        self._calculate_drawdown_metrics(pnls)
        self.metrics['profit_factor'] = self._calculate_profit_factor(pnls)
        
        logger.debug(f"Max drawdown: {self.metrics['max_drawdown']:.2f}")
        logger.debug(f"Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}")
        logger.debug(f"Profit factor: {self.metrics['profit_factor']:.2f}")
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from list of P&Ls"""
        if not pnls:
            return 0.0
            
        cumulative = np.cumsum([0] + pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return abs(min(drawdowns))
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio from returns"""
        if not returns:
            return 0.0
            
        # Assuming daily returns, annualize with sqrt(252)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
            
        return mean_return / std_return * np.sqrt(252)
        
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate annualized Sortino ratio from returns"""
        if not returns:
            return 0.0
            
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0
            
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
            
        return mean_return / downside_std * np.sqrt(252)
        
    def _calculate_drawdown_metrics(self, pnls: List[float]) -> None:
        """Calculate maximum drawdown duration and recovery period"""
        if not pnls:
            return
            
        cumulative = np.cumsum([0] + pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        
        # Find the maximum drawdown period
        drawdown_periods = np.where(drawdowns < 0)[0]
        if len(drawdown_periods) > 0:
            max_dd_end = drawdown_periods[np.argmin(drawdowns)]
            max_dd_start = np.where(running_max[:max_dd_end] == running_max[max_dd_end])[0][-1]
            self.metrics['max_drawdown_duration'] = max_dd_end - max_dd_start
            
            # Find recovery period (time to recover from max drawdown)
            recovery_end = next((i for i in range(max_dd_end, len(cumulative)) if cumulative[i] >= running_max[max_dd_end]), len(cumulative))
            self.metrics['recovery_period'] = recovery_end - max_dd_end
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if not pnls:
            return 0.0
            
        gross_profits = sum(pnl for pnl in pnls if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in pnls if pnl < 0))
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0
            
        return gross_profits / gross_losses

class Backtester:
    def __init__(self, strategy: Strategy, initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_trade = None
        self.logger = logging.getLogger(__name__)
        
    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            BacktestResult object containing trades and metrics
        """
        logger.info("Starting backtest...")
        result = BacktestResult()
        
        for i in range(len(data)):
            current_slice = data.iloc[:i+1]
            current_bar = current_slice.iloc[-1]
            
            # Generate signals
            signals = self.strategy.analyze(current_slice)
            if signals:
                logger.info(f"Signal generated at {current_bar.name}: {signals[0]}")
            
            # Process signals
            for signal in signals:
                if signal['Type'] == 'BUY' and self.current_trade is None:
                    # Open long position
                    self.current_trade = Trade(
                        entry_price=signal['Price'],
                        entry_date=current_bar.name
                    )
                    logger.info(f"Opening long position at {signal['Price']:.2f}")
                    
                elif signal['Type'] == 'SELL' and self.current_trade is not None:
                    # Close long position
                    self.current_trade.close(
                        exit_price=signal['Price'],
                        exit_date=current_bar.name
                    )
                    result.add_trade(self.current_trade)
                    logger.info(f"Closing long position at {signal['Price']:.2f}, "
                              f"P&L: {self.current_trade.pnl:.2f}")
                    self.current_trade = None
        
        # Close any open trades at the end
        if self.current_trade is not None:
            self.current_trade.close(
                exit_price=data.iloc[-1]['Close'],
                exit_date=data.index[-1]
            )
            result.add_trade(self.current_trade)
            logger.info(f"Closing final position at {data.iloc[-1]['Close']:.2f}")
        
        # Calculate performance metrics
        result.calculate_metrics()
        logger.info("Backtest completed")
        
        return result
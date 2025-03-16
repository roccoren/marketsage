"""
Performance metrics calculation for backtesting.
Calculates various performance metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from decimal import Decimal
import math
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculates performance metrics for backtesting results.
    Includes return metrics, risk metrics, and trade metrics.
    """
    
    @staticmethod
    def calculate_metrics(equity_curve: Union[List[Dict], pd.DataFrame], 
                          trades: Optional[List[Dict]] = None,
                          risk_free_rate: float = 0.0,
                          benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate performance metrics from equity curve and trades.
        
        Args:
            equity_curve: List of equity history dictionaries or DataFrame
            trades: List of trade dictionaries
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dict with performance metrics
        """
        # Convert equity curve to DataFrame if it's a list
        if isinstance(equity_curve, list):
            if not equity_curve:
                return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
                
            equity_df = pd.DataFrame(equity_curve)
            
            # Handle different timestamp column names
            timestamp_col = None
            for col in ['timestamp', 'date', 'time']:
                if col in equity_df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col:
                equity_df[timestamp_col] = pd.to_datetime(equity_df[timestamp_col])
                equity_df.set_index(timestamp_col, inplace=True)
                equity_df.sort_index(inplace=True)
        else:
            equity_df = equity_curve.copy()
            
            # If DataFrame doesn't have a datetime index, try to set it
            if not isinstance(equity_df.index, pd.DatetimeIndex):
                for col in ['timestamp', 'date', 'time']:
                    if col in equity_df.columns:
                        equity_df[col] = pd.to_datetime(equity_df[col])
                        equity_df.set_index(col, inplace=True)
                        equity_df.sort_index(inplace=True)
                        break
        
        # Ensure we have an 'equity' column
        if 'equity' not in equity_df.columns:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
            
        # Convert equity to numeric if it's not already
        equity_df['equity'] = pd.to_numeric(equity_df['equity'])
            
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate log returns safely
        try:
            equity_df['log_returns'] = np.log(equity_df['equity'] / equity_df['equity'].shift(1))
        except (TypeError, ValueError):
            # Fallback if log calculation fails
            equity_df['log_returns'] = np.log1p(equity_df['returns'])
            
        equity_df['cum_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Return metrics
        metrics["total_return"] = PerformanceMetrics.calculate_total_return(equity_df)
        metrics["annualized_return"] = PerformanceMetrics.calculate_annualized_return(equity_df)
        metrics["daily_returns_mean"] = PerformanceMetrics.calculate_daily_returns(equity_df).mean() if len(equity_df) > 1 else 0.0
        metrics["monthly_returns_mean"] = PerformanceMetrics.calculate_monthly_returns(equity_df).mean() if len(equity_df) > 1 else 0.0
        
        # Risk metrics
        metrics["max_drawdown"] = PerformanceMetrics.calculate_max_drawdown(equity_df)
        metrics["sharpe_ratio"] = PerformanceMetrics.calculate_sharpe_ratio(equity_df, risk_free_rate)
        metrics["sortino_ratio"] = PerformanceMetrics.calculate_sortino_ratio(equity_df, risk_free_rate)
        metrics["calmar_ratio"] = PerformanceMetrics.calculate_calmar_ratio(equity_df)
        metrics["volatility"] = PerformanceMetrics.calculate_volatility(equity_df)
        metrics["var_95"] = PerformanceMetrics.calculate_var(equity_df, 0.95)
        metrics["cvar_95"] = PerformanceMetrics.calculate_cvar(equity_df, 0.95)
        
        # Trade metrics (only if trades are provided)
        if trades:
            metrics["total_trades"] = len(trades)
            metrics["winning_trades"] = PerformanceMetrics.calculate_winning_trades(trades)
            metrics["losing_trades"] = PerformanceMetrics.calculate_losing_trades(trades)
            metrics["win_rate"] = PerformanceMetrics.calculate_win_rate(trades)
            metrics["profit_factor"] = PerformanceMetrics.calculate_profit_factor(trades)
            metrics["average_win"] = PerformanceMetrics.calculate_average_win(trades)
            metrics["average_loss"] = PerformanceMetrics.calculate_average_loss(trades)
            metrics["average_trade"] = PerformanceMetrics.calculate_average_trade(trades)
            metrics["expectancy"] = PerformanceMetrics.calculate_expectancy(trades)
            metrics["average_holding_period"] = PerformanceMetrics.calculate_average_holding_period(trades)
        else:
            # Default values for trade metrics
            metrics["total_trades"] = 0
            metrics["winning_trades"] = 0
            metrics["losing_trades"] = 0
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["average_win"] = 0.0
            metrics["average_loss"] = 0.0
            metrics["average_trade"] = 0.0
            metrics["expectancy"] = 0.0
            metrics["average_holding_period"] = 0.0
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics["alpha"] = PerformanceMetrics.calculate_alpha(equity_df, benchmark_returns, risk_free_rate)
            metrics["beta"] = PerformanceMetrics.calculate_beta(equity_df, benchmark_returns)
            metrics["information_ratio"] = PerformanceMetrics.calculate_information_ratio(equity_df, benchmark_returns)
            metrics["tracking_error"] = PerformanceMetrics.calculate_tracking_error(equity_df, benchmark_returns)
        
        return metrics
    
    @staticmethod
    def calculate_total_return(equity_df: pd.DataFrame) -> float:
        """Calculate total return."""
        if len(equity_df) < 2:
            return 0.0
        
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        
        return float((final_equity / initial_equity) - 1.0)
    
    @staticmethod
    def calculate_annualized_return(equity_df: pd.DataFrame) -> float:
        """Calculate annualized return."""
        if len(equity_df) < 2:
            return 0.0
        
        total_return = PerformanceMetrics.calculate_total_return(equity_df)
        
        # Calculate years
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        days = (end_date - start_date).days
        years = days / 365.0
        
        if years < 0.01:  # Less than ~3.65 days
            return 0.0
            
        # Handle negative returns to avoid complex numbers
        if total_return < -1.0:
            # For catastrophic losses, just return the total return
            return float(total_return)
            
        try:
            # Use safe calculation to avoid complex numbers
            if total_return >= 0:
                return float(((1.0 + total_return) ** (1.0 / years)) - 1.0)
            else:
                # For negative returns, use a different approach
                return float(-1.0 * ((1.0 - abs(total_return)) ** (1.0 / years) - 1.0))
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"Error calculating annualized return: {e}. Total return: {total_return}, Years: {years}")
            # Fallback to simple annualized return
            return float(total_return / years)
    
    @staticmethod
    def calculate_daily_returns(equity_df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns."""
        if 'returns' not in equity_df.columns or len(equity_df) < 2:
            return pd.Series()
            
        # Resample to daily if needed
        if equity_df.index.inferred_freq != 'D':
            daily_equity = equity_df['equity'].resample('D').last().dropna()
            daily_returns = daily_equity.pct_change().dropna()
        else:
            daily_returns = equity_df['returns'].dropna()
            
        return daily_returns
    
    @staticmethod
    def calculate_monthly_returns(equity_df: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns."""
        if len(equity_df) < 2:
            return pd.Series()
            
        # Resample to monthly
        monthly_equity = equity_df['equity'].resample('ME').last().dropna()
        monthly_returns = monthly_equity.pct_change().dropna()
            
        return monthly_returns
    
    @staticmethod
    def calculate_max_drawdown(equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if len(equity_df) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_df['equity'].cummax()
        
        # Calculate drawdown
        drawdown = (equity_df['equity'] / running_max) - 1.0
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return float(abs(max_drawdown))
    
    @staticmethod
    def calculate_sharpe_ratio(equity_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        daily_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Calculate excess returns
        excess_returns = daily_returns - (risk_free_rate / 252.0)  # Daily risk-free rate
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252.0)  # Annualized
        
        return float(sharpe)
    
    @staticmethod
    def calculate_sortino_ratio(equity_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        daily_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Calculate excess returns
        excess_returns = daily_returns - (risk_free_rate / 252.0)  # Daily risk-free rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        # Calculate Sortino ratio
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252.0)  # Annualized
        
        return float(sortino)
    
    @staticmethod
    def calculate_calmar_ratio(equity_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio."""
        if len(equity_df) < 30:
            return 0.0
            
        # Get annualized return
        annualized_return = PerformanceMetrics.calculate_annualized_return(equity_df)
        
        # Get maximum drawdown
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity_df)
        
        if max_drawdown == 0:
            return 0.0
            
        # Calculate Calmar ratio
        calmar = annualized_return / max_drawdown
        
        return float(calmar)
    
    @staticmethod
    def calculate_volatility(equity_df: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        if 'returns' not in equity_df.columns or len(equity_df) < 2:
            return 0.0
            
        # Get daily returns
        daily_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Calculate annualized volatility
        volatility = daily_returns.std() * np.sqrt(252.0)
        
        return float(volatility)
    
    @staticmethod
    def calculate_var(equity_df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        daily_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Calculate VaR
        var = np.percentile(daily_returns, 100 * (1 - confidence))
        
        return float(abs(var))
    
    @staticmethod
    def calculate_cvar(equity_df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        daily_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Calculate VaR
        var = np.percentile(daily_returns, 100 * (1 - confidence))
        
        # Calculate CVaR (Expected Shortfall)
        cvar = daily_returns[daily_returns <= var].mean()
        
        return float(abs(cvar))
    
    @staticmethod
    def calculate_winning_trades(trades: List[Dict]) -> int:
        """Calculate number of winning trades."""
        winning_trades = [t for t in trades if t.get('type') == 'CLOSE' and t.get('pnl', 0) > 0]
        return len(winning_trades)
    
    @staticmethod
    def calculate_losing_trades(trades: List[Dict]) -> int:
        """Calculate number of losing trades."""
        losing_trades = [t for t in trades if t.get('type') == 'CLOSE' and t.get('pnl', 0) <= 0]
        return len(losing_trades)
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """Calculate win rate."""
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        
        if not closed_trades:
            return 0.0
            
        winning_trades = PerformanceMetrics.calculate_winning_trades(trades)
        
        return float(winning_trades / len(closed_trades))
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor."""
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        
        if not closed_trades:
            return 0.0
            
        gross_profit = sum(float(t.get('pnl', 0)) for t in closed_trades if t.get('pnl', 0) > 0)
        gross_loss = sum(abs(float(t.get('pnl', 0))) for t in closed_trades if t.get('pnl', 0) < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_average_win(trades: List[Dict]) -> float:
        """Calculate average win."""
        winning_trades = [t for t in trades if t.get('type') == 'CLOSE' and t.get('pnl', 0) > 0]
        
        if not winning_trades:
            return 0.0
            
        total_profit = sum(float(t.get('pnl', 0)) for t in winning_trades)
        
        return total_profit / len(winning_trades)
    
    @staticmethod
    def calculate_average_loss(trades: List[Dict]) -> float:
        """Calculate average loss."""
        losing_trades = [t for t in trades if t.get('type') == 'CLOSE' and t.get('pnl', 0) < 0]
        
        if not losing_trades:
            return 0.0
            
        total_loss = sum(float(t.get('pnl', 0)) for t in losing_trades)
        
        return total_loss / len(losing_trades)
    
    @staticmethod
    def calculate_average_trade(trades: List[Dict]) -> float:
        """Calculate average trade P&L."""
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE']
        
        if not closed_trades:
            return 0.0
            
        total_pnl = sum(float(t.get('pnl', 0)) for t in closed_trades)
        
        return total_pnl / len(closed_trades)
    
    @staticmethod
    def calculate_expectancy(trades: List[Dict]) -> float:
        """Calculate expectancy (average trade * win rate)."""
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        average_win = PerformanceMetrics.calculate_average_win(trades)
        average_loss = PerformanceMetrics.calculate_average_loss(trades)
        
        return (win_rate * average_win) + ((1 - win_rate) * average_loss)
    
    @staticmethod
    def calculate_average_holding_period(trades: List[Dict]) -> float:
        """Calculate average holding period in days."""
        # Pair open and close trades
        open_trades = {t['position_id']: t for t in trades if t.get('type') == 'OPEN'}
        close_trades = [t for t in trades if t.get('type') == 'CLOSE']
        
        if not close_trades:
            return 0.0
            
        holding_periods = []
        
        for close_trade in close_trades:
            position_id = close_trade.get('position_id')
            
            if position_id in open_trades:
                open_time = pd.to_datetime(open_trades[position_id]['timestamp'])
                close_time = pd.to_datetime(close_trade['timestamp'])
                
                holding_period = (close_time - open_time).total_seconds() / (24 * 60 * 60)  # Convert to days
                holding_periods.append(holding_period)
                
        if not holding_periods:
            return 0.0
            
        return sum(holding_periods) / len(holding_periods)
    
    @staticmethod
    def calculate_alpha(equity_df: pd.DataFrame, 
                        benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.0) -> float:
        """Calculate Jensen's Alpha."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        strategy_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Align benchmark returns with strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return 0.0
            
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate beta
        beta = PerformanceMetrics.calculate_beta(equity_df, benchmark_returns)
        
        # Calculate alpha
        daily_rf = risk_free_rate / 252.0
        alpha = (strategy_returns.mean() - daily_rf) - beta * (benchmark_returns.mean() - daily_rf)
        
        # Annualize alpha
        alpha_annualized = alpha * 252.0
        
        return float(alpha_annualized)
    
    @staticmethod
    def calculate_beta(equity_df: pd.DataFrame, benchmark_returns: pd.Series) -> float:
        """Calculate beta (systematic risk)."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        strategy_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Align benchmark returns with strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return 0.0
            
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate covariance and variance
        covariance = strategy_returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        
        if variance == 0:
            return 0.0
            
        beta = covariance / variance
        
        return float(beta)
    
    @staticmethod
    def calculate_information_ratio(equity_df: pd.DataFrame, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        strategy_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Align benchmark returns with strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return 0.0
            
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate tracking error
        tracking_error = PerformanceMetrics.calculate_tracking_error(equity_df, benchmark_returns)
        
        if tracking_error == 0:
            return 0.0
            
        # Calculate information ratio
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = (excess_returns.mean() * 252.0) / (tracking_error * np.sqrt(252.0))
        
        return float(information_ratio)
    
    @staticmethod
    def calculate_tracking_error(equity_df: pd.DataFrame, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error."""
        if 'returns' not in equity_df.columns or len(equity_df) < 30:
            return 0.0
            
        # Get daily returns
        strategy_returns = PerformanceMetrics.calculate_daily_returns(equity_df)
        
        # Align benchmark returns with strategy returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return 0.0
            
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate tracking error
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        return float(tracking_error)
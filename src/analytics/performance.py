"""
Performance analytics module for calculating and visualizing strategy performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..config.settings import (
    PLOT_SETTINGS,
    PERFORMANCE_METRICS,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR
)

class PerformanceAnalytics:
    """Class for calculating and visualizing strategy performance metrics"""
    
    def __init__(
        self,
        returns: pd.Series,
        positions: Optional[Dict[str, pd.Series]] = None,
        benchmark_returns: Optional[pd.Series] = None
    ):
        """
        Initialize the performance analytics
        
        Args:
            returns: Series of strategy returns
            positions: Optional dictionary of position series for each symbol
            benchmark_returns: Optional series of benchmark returns
        """
        self.returns = returns
        self.positions = positions or {}
        self.benchmark_returns = benchmark_returns
        
        # Set plot style
        plt.style.use(PLOT_SETTINGS['style'])
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Calculate basic return metrics
        metrics['total_return'] = self._calculate_total_return()
        metrics['annual_return'] = self._calculate_annual_return()
        metrics['daily_return'] = self.returns.mean()
        metrics['daily_volatility'] = self.returns.std()
        
        # Calculate risk metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio()
        metrics['sortino_ratio'] = self._calculate_sortino_ratio()
        metrics['max_drawdown'] = self._calculate_max_drawdown()
        metrics['calmar_ratio'] = self._calculate_calmar_ratio()
        
        # Calculate trading metrics
        metrics['win_rate'] = self._calculate_win_rate()
        metrics['profit_factor'] = self._calculate_profit_factor()
        metrics['avg_win'] = self._calculate_avg_win()
        metrics['avg_loss'] = self._calculate_avg_loss()
        
        # Calculate relative metrics if benchmark is provided
        if self.benchmark_returns is not None:
            metrics['information_ratio'] = self._calculate_information_ratio()
            metrics['alpha'] = self._calculate_alpha()
            metrics['beta'] = self._calculate_beta()
            
        return metrics
    
    def _calculate_total_return(self) -> float:
        """Calculate total return"""
        return (1 + self.returns).prod() - 1
    
    def _calculate_annual_return(self) -> float:
        """Calculate annual return"""
        years = (self.returns.index[-1] - self.returns.index[0]).days / 365
        return (1 + self._calculate_total_return()) ** (1 / years) - 1
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = self.returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        excess_returns = self.returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / downside_std if downside_std != 0 else float('inf')
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        max_dd = self._calculate_max_drawdown()
        return self._calculate_annual_return() / abs(max_dd) if max_dd != 0 else float('inf')
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        return (self.returns > 0).mean()
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profits = self.returns[self.returns > 0].sum()
        gross_losses = abs(self.returns[self.returns < 0].sum())
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def _calculate_avg_win(self) -> float:
        """Calculate average win"""
        winning_trades = self.returns[self.returns > 0]
        return winning_trades.mean() if not winning_trades.empty else 0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average loss"""
        losing_trades = self.returns[self.returns < 0]
        return losing_trades.mean() if not losing_trades.empty else 0
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio"""
        excess_returns = self.returns - self.benchmark_returns
        return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_alpha(self) -> float:
        """Calculate alpha"""
        beta = self._calculate_beta()
        excess_returns = self.returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        benchmark_excess = self.benchmark_returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        return excess_returns.mean() - beta * benchmark_excess.mean()
    
    def _calculate_beta(self) -> float:
        """Calculate beta"""
        covariance = self.returns.cov(self.benchmark_returns)
        benchmark_variance = self.benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    def plot_returns(self, save_path: Optional[str] = None) -> None:
        """
        Plot cumulative returns
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=PLOT_SETTINGS['figsize'])
        
        # Plot strategy returns
        cumulative_returns = (1 + self.returns).cumprod()
        plt.plot(
            cumulative_returns.index,
            cumulative_returns.values,
            label='Strategy',
            color=PLOT_SETTINGS['colors']['returns']
        )
        
        # Plot benchmark returns if available
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            plt.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label='Benchmark',
                color='gray',
                alpha=0.5
            )
        
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['dpi'])
        plt.close()
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        """
        Plot drawdown
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=PLOT_SETTINGS['figsize'])
        
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        
        plt.plot(
            drawdowns.index,
            drawdowns.values,
            label='Drawdown',
            color='red'
        )
        
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['dpi'])
        plt.close()
    
    def plot_monthly_returns(self, save_path: Optional[str] = None) -> None:
        """
        Plot monthly returns heatmap
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=PLOT_SETTINGS['figsize'])
        
        # Resample returns to monthly frequency
        monthly_returns = self.returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_table = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).mean().unstack()
        
        # Plot heatmap
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Monthly Return'}
        )
        
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['dpi'])
        plt.close()
    
    def plot_position_weights(self, save_path: Optional[str] = None) -> None:
        """
        Plot position weights over time
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.positions:
            return
            
        plt.figure(figsize=PLOT_SETTINGS['figsize'])
        
        # Combine position series into DataFrame
        position_df = pd.DataFrame(self.positions)
        
        # Plot stacked area chart
        position_df.plot(
            kind='area',
            stacked=True,
            colormap='Set3'
        )
        
        plt.title('Position Weights Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=PLOT_SETTINGS['dpi'], bbox_inches='tight')
        plt.close() 
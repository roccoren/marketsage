"""
Visualization module for backtesting results.
Provides various visualization tools for analyzing backtest performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualization:
    """
    Visualization tools for backtesting results.
    Provides methods for creating various performance charts.
    """
    
    def __init__(self):
        """Initialize the visualization module."""
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    @staticmethod
    def plot_equity_curve(equity_curve: pd.DataFrame, 
                          benchmark: Optional[pd.DataFrame] = None,
                          title: str = "Equity Curve",
                          figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot equity curve with optional benchmark.
        
        Args:
            equity_curve: DataFrame with equity history
            benchmark: Optional benchmark equity curve
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve['equity'], 
                label='Strategy', color='#1f77b4', linewidth=2)
        
        # Add initial capital as horizontal line
        initial_capital = equity_curve['equity'].iloc[0]
        ax.axhline(y=initial_capital, color='#ff7f0e', linestyle='--', 
                   label=f'Initial Capital (${initial_capital:,.2f})')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Ensure benchmark has a datetime index
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                if 'timestamp' in benchmark.columns:
                    benchmark = benchmark.set_index('timestamp')
                elif 'date' in benchmark.columns:
                    benchmark = benchmark.set_index('date')
                    
            # Normalize benchmark to start at the same value as the strategy
            benchmark_normalized = benchmark * (initial_capital / benchmark.iloc[0])
            
            # Plot benchmark
            ax.plot(benchmark_normalized.index, benchmark_normalized, 
                    label='Benchmark', color='#2ca02c', linewidth=1.5, alpha=0.7)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_drawdown(equity_curve: pd.DataFrame,
                      title: str = "Drawdown",
                      figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot drawdown chart.
        
        Args:
            equity_curve: DataFrame with equity history
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate drawdown
        running_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] / running_max - 1.0) * 100
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_monthly_returns(equity_curve: pd.DataFrame,
                             title: str = "Monthly Returns",
                             figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_curve: DataFrame with equity history
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate returns if not already present
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            
        # Resample to monthly returns
        monthly_returns = equity_curve['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ).to_frame()
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        pivot = monthly_returns.pivot_table(
            index='year', 
            columns='month', 
            values='returns'
        )
        
        # Replace month numbers with month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot.columns = [month_names[m] for m in pivot.columns]
        
        # Create a custom colormap (green for positive, red for negative)
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(
            pivot * 100,  # Convert to percentage
            annot=True,
            fmt=".1f",
            cmap=cmap,
            center=0,
            linewidths=1,
            cbar_kws={'label': 'Monthly Return (%)'},
            ax=ax
        )
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_trade_distribution(trades: List[Dict],
                                title: str = "Trade Distribution",
                                figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot trade P&L distribution.
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract closed trades with P&L
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE' and 'pnl' in t]
        
        if not closed_trades:
            logger.warning("No closed trades with P&L data available for distribution chart")
            ax.text(0.5, 0.5, "No trade data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            return fig
            
        # Extract P&L values
        pnl_values = [float(t['pnl']) for t in closed_trades]
        
        # Plot histogram
        n, bins, patches = ax.hist(pnl_values, bins=30, alpha=0.7, color='#1f77b4')
        
        # Color negative and positive bars differently
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor('#d62728')  # Red for negative
            else:
                patches[i].set_facecolor('#2ca02c')  # Green for positive
                
        # Add a vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('P&L ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with dollar signs
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_rolling_sharpe(equity_curve: pd.DataFrame,
                            window: int = 252,
                            risk_free_rate: float = 0.0,
                            title: str = "Rolling Sharpe Ratio",
                            figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            equity_curve: DataFrame with equity history
            window: Rolling window size in days
            risk_free_rate: Risk-free rate for Sharpe calculation
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate returns if not already present
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            
        # Calculate daily excess returns
        daily_rf = risk_free_rate / 252.0
        excess_returns = equity_curve['returns'] - daily_rf
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = (
            excess_returns.rolling(window=window).mean() / 
            excess_returns.rolling(window=window).std()
        ) * np.sqrt(252.0)  # Annualize
        
        # Plot rolling Sharpe ratio
        ax.plot(rolling_sharpe.index, rolling_sharpe, color='#1f77b4', linewidth=1.5)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add horizontal lines at 1 and 2 (common Sharpe ratio thresholds)
        ax.axhline(y=1, color='#ff7f0e', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='#2ca02c', linestyle='--', alpha=0.5, label='Sharpe = 2')
        
        # Format plot
        ax.set_title(f"{title} ({window}-day)", fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_underwater(equity_curve: pd.DataFrame,
                        title: str = "Underwater Plot",
                        figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot underwater chart (continuous drawdown).
        
        Args:
            equity_curve: DataFrame with equity history
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate drawdown
        running_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] / running_max - 1.0) * 100
        
        # Plot underwater chart
        ax.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
        
        # Add horizontal lines at -5%, -10%, -20% (common drawdown thresholds)
        ax.axhline(y=-5, color='#ff7f0e', linestyle='--', alpha=0.5, label='-5%')
        ax.axhline(y=-10, color='#ff7f0e', linestyle='--', alpha=0.5, label='-10%')
        ax.axhline(y=-20, color='#d62728', linestyle='--', alpha=0.5, label='-20%')
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_returns_histogram(equity_curve: pd.DataFrame,
                               title: str = "Returns Distribution",
                               figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot histogram of returns.
        
        Args:
            equity_curve: DataFrame with equity history
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate returns if not already present
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            
        # Remove NaN values
        returns = equity_curve['returns'].dropna()
        
        # Convert to percentage
        returns_pct = returns * 100
        
        # Plot histogram
        n, bins, patches = ax.hist(returns_pct, bins=50, alpha=0.7, color='#1f77b4', density=True)
        
        # Color negative and positive bars differently
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor('#d62728')  # Red for negative
            else:
                patches[i].set_facecolor('#2ca02c')  # Green for positive
                
        # Add a vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add normal distribution curve
        mu = returns_pct.mean()
        sigma = returns_pct.std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        ax.plot(x, y, 'k--', linewidth=1.5, label='Normal Distribution')
        
        # Format plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Daily Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics as text
        stats_text = (
            f"Mean: {mu:.2f}%\n"
            f"Std Dev: {sigma:.2f}%\n"
            f"Skew: {returns_pct.skew():.2f}\n"
            f"Kurtosis: {returns_pct.kurtosis():.2f}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def plot_rolling_volatility(equity_curve: pd.DataFrame,
                                window: int = 252,
                                title: str = "Rolling Volatility",
                                figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot rolling volatility.
        
        Args:
            equity_curve: DataFrame with equity history
            window: Rolling window size in days
            title: Plot title
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Calculate returns if not already present
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            
        # Calculate rolling volatility (annualized)
        rolling_vol = equity_curve['returns'].rolling(window=window).std() * np.sqrt(252) * 100
        
        # Plot rolling volatility
        ax.plot(rolling_vol.index, rolling_vol, color='#1f77b4', linewidth=1.5)
        
        # Format plot
        ax.set_title(f"{title} ({window}-day)", fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def save_figure(fig: Figure, 
                    output_path: Union[str, Path],
                    dpi: int = 100) -> None:
        """
        Save a figure to a file.
        
        Args:
            fig: Matplotlib figure to save
            output_path: Path to save the figure
            dpi: Resolution in dots per inch
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        
        logger.info(f"Saved figure to {output_path}")
        
    @staticmethod
    def generate_all_charts(equity_curve: pd.DataFrame,
                            trades: List[Dict],
                            output_dir: Union[str, Path],
                            benchmark: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
        """
        Generate all available charts and save them to files.
        
        Args:
            equity_curve: DataFrame with equity history
            trades: List of trade dictionaries
            output_dir: Directory to save charts
            benchmark: Optional benchmark equity curve
            
        Returns:
            Dict mapping chart name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chart_paths = {}
        
        # Ensure equity_curve has a datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if 'timestamp' in equity_curve.columns:
                equity_curve = equity_curve.set_index('timestamp')
            elif 'date' in equity_curve.columns:
                equity_curve = equity_curve.set_index('date')
                
        # Generate and save equity curve
        fig = Visualization.plot_equity_curve(equity_curve, benchmark)
        path = output_dir / "equity_curve.png"
        Visualization.save_figure(fig, path)
        chart_paths["equity_curve"] = path
        
        # Generate and save drawdown
        fig = Visualization.plot_drawdown(equity_curve)
        path = output_dir / "drawdown.png"
        Visualization.save_figure(fig, path)
        chart_paths["drawdown"] = path
        
        # Generate and save underwater plot
        fig = Visualization.plot_underwater(equity_curve)
        path = output_dir / "underwater.png"
        Visualization.save_figure(fig, path)
        chart_paths["underwater"] = path
        
        # Generate and save monthly returns
        fig = Visualization.plot_monthly_returns(equity_curve)
        path = output_dir / "monthly_returns.png"
        Visualization.save_figure(fig, path)
        chart_paths["monthly_returns"] = path
        
        # Generate and save returns histogram
        fig = Visualization.plot_returns_histogram(equity_curve)
        path = output_dir / "returns_histogram.png"
        Visualization.save_figure(fig, path)
        chart_paths["returns_histogram"] = path
        
        # Generate and save rolling volatility
        fig = Visualization.plot_rolling_volatility(equity_curve)
        path = output_dir / "rolling_volatility.png"
        Visualization.save_figure(fig, path)
        chart_paths["rolling_volatility"] = path
        
        # Generate and save rolling Sharpe ratio
        fig = Visualization.plot_rolling_sharpe(equity_curve)
        path = output_dir / "rolling_sharpe.png"
        Visualization.save_figure(fig, path)
        chart_paths["rolling_sharpe"] = path
        
        # Generate and save trade distribution
        if trades:
            fig = Visualization.plot_trade_distribution(trades)
            path = output_dir / "trade_distribution.png"
            Visualization.save_figure(fig, path)
            chart_paths["trade_distribution"] = path
            
        logger.info(f"Generated {len(chart_paths)} charts in {output_dir}")
        
        return chart_paths 
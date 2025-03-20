"""
Visualization utilities for strategy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path
from ..config.settings import OUTPUT_DIR, PLOT_SETTINGS

def plot_strategy_performance(returns: pd.Series, title: str = "Strategy Performance"):
    """Plot strategy performance metrics"""
    plt.style.use(PLOT_SETTINGS['style'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    
    # Plot cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    ax1.plot(cumulative_returns.index, cumulative_returns.values, color=PLOT_SETTINGS['colors']['returns'])
    ax1.set_title(f"{title} - Cumulative Returns")
    ax1.grid(True)
    
    # Plot drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    ax2.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)
    ax2.set_title("Drawdown")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_performance.png", format=PLOT_SETTINGS['save_format'])
    plt.close()

def plot_signals_and_indicators(
    data: pd.DataFrame,
    signals: pd.Series,
    title: str = "Trading Signals and Indicators"
):
    """Plot trading signals and technical indicators"""
    plt.style.use(PLOT_SETTINGS['style'])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    
    # Plot price and moving averages
    ax1.plot(data.index, data['Close'], label='Price', color='black')
    ax1.plot(data.index, data['ma10'], label='MA10', color='blue')
    ax1.plot(data.index, data['ma20'], label='MA20', color='orange')
    ax1.plot(data.index, data['ma50'], label='MA50', color='red')
    ax1.set_title(f"{title} - Price and Moving Averages")
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI
    ax2.plot(data.index, data['rsi'], color='purple')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title("RSI")
    ax2.grid(True)
    
    # Plot signals
    ax3.plot(signals.index, signals.values, color=PLOT_SETTINGS['colors']['signals'])
    ax3.set_title("Trading Signals")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signals_and_indicators.png", format=PLOT_SETTINGS['save_format'])
    plt.close()

def plot_portfolio_weights(positions: Dict[str, pd.Series], title: str = "Portfolio Weights"):
    """Plot portfolio weights over time"""
    plt.style.use(PLOT_SETTINGS['style'])
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    
    # Combine all position series into a DataFrame
    weights_df = pd.DataFrame(positions)
    
    # Plot stacked area chart
    weights_df.plot(kind='area', stacked=True, ax=ax)
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "portfolio_weights.png", format=PLOT_SETTINGS['save_format'])
    plt.close()

def plot_correlation_matrix(returns: Dict[str, pd.Series], title: str = "Returns Correlation Matrix"):
    """Plot correlation matrix of asset returns"""
    plt.style.use(PLOT_SETTINGS['style'])
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    
    # Calculate correlation matrix
    returns_df = pd.DataFrame(returns)
    corr_matrix = returns_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", format=PLOT_SETTINGS['save_format'])
    plt.close() 
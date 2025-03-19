"""
Utilities for formatting and displaying backtest results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def display_results(symbol: str, exchange: str, result: Any, show_plots: bool = True,
                   save_plots: bool = True, output_dir: Optional[str] = None) -> None:
    """
    Display and optionally save backtest results
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    exchange : str
        Exchange name
    result : Any
        Backtest result object with metrics attribute
    show_plots : bool, optional
        Whether to display plots, by default True
    save_plots : bool, optional
        Whether to save plots, by default True
    output_dir : str, optional
        Output directory for plots, by default "output/backtest/plots"
    """
    metrics = result.metrics
    
    # Print summary metrics
    print(f"\n{'=' * 50}")
    print(f"BACKTEST RESULTS: {symbol} ({exchange})")
    print(f"{'=' * 50}")
    
    # Performance metrics
    print(f"\nPERFORMANCE METRICS:")
    print(f"Net Profit:          {metrics.get('net_profit', 0):.2f} ({metrics.get('net_profit_pct', 0):.2f}%)")
    print(f"Buy & Hold Return:   {metrics.get('buy_hold_return', 0):.2f} ({metrics.get('buy_hold_return_pct', 0):.2f}%)")
    print(f"Alpha:               {metrics.get('alpha', 0):.4f}")
    print(f"Beta:                {metrics.get('beta', 0):.4f}")
    
    # Risk metrics
    print(f"\nRISK METRICS:")
    print(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Sortino Ratio:       {metrics.get('sortino_ratio', 0):.4f}")
    print(f"Maximum Drawdown:    {metrics.get('max_drawdown', 0):.2f}% over {metrics.get('max_drawdown_duration', 0)} days")
    print(f"Calmar Ratio:        {metrics.get('calmar_ratio', 0):.4f}")
    
    # Trade metrics
    print(f"\nTRADE METRICS:")
    print(f"Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"Win Rate:            {metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"Average Win:         {metrics.get('avg_win', 0):.2f}")
    print(f"Average Loss:        {metrics.get('avg_loss', 0):.2f}")
    print(f"Profit Factor:       {metrics.get('profit_factor', 0):.4f}")
    print(f"Recovery Factor:     {metrics.get('recovery_factor', 0):.4f}")
    print(f"Expectancy:          {metrics.get('expectancy', 0):.4f}")
    
    # Strategy Parameters
    if hasattr(result, 'params') and result.params:
        print(f"\nSTRATEGY PARAMETERS:")
        for k, v in result.params.items():
            print(f"{k}: {v}")

    # Generate plots if requested
    if show_plots or save_plots:
        try:
            generate_plots(symbol, exchange, result, show_plots, save_plots, output_dir)
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
    
    print(f"\n{'=' * 50}\n")


def generate_plots(symbol: str, exchange: str, result: Any, show_plots: bool = True,
                  save_plots: bool = True, output_dir: Optional[str] = None) -> None:
    """
    Generate performance plots from backtest results
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    exchange : str
        Exchange name
    result : Any
        Backtest result object
    show_plots : bool, optional
        Whether to display plots, by default True
    save_plots : bool, optional
        Whether to save plots, by default True
    output_dir : str, optional
        Output directory for plots, by default "output/backtest/plots"
    """
    try:
        # Setup plot output directory
        if save_plots:
            output_path = Path(output_dir) if output_dir else Path("output/backtest/plots")
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Get equity curve and trade data
        if hasattr(result, 'equity_curve'):
            # Ensure equity_curve is a DataFrame
            if isinstance(result.equity_curve, pd.Series):
                equity_curve = pd.DataFrame({'equity': result.equity_curve})
            else:
                equity_curve = result.equity_curve
        else:
            logger.warning("No equity curve data available for plotting")
            return
        
        trades = result.trades if hasattr(result, 'trades') else None
        signals = result.signals if hasattr(result, 'signals') else None
        
        if equity_curve is None or len(equity_curve) == 0:
            logger.warning("No equity curve data available for plotting")
            return
        
        # Set plot style and figure size
        plt.style.use('seaborn-v0_8-darkgrid')
        
        try:
            # 1. Equity Curve and Drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve['equity'], label='Strategy', linewidth=2)
            
            # If we have a benchmark, plot it
            if 'benchmark' in equity_curve.columns:
                ax1.plot(equity_curve.index, equity_curve['benchmark'], label='Buy & Hold', 
                        linewidth=1.5, linestyle='--', alpha=0.7)
            
            # Add markers for trades if available
            if trades is not None and len(trades) > 0:
                try:
                    trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades
                    logger.info(f"Trade columns: {trades_df.columns.tolist()}")
                    
                    # Check for entry/exit dates
                    if 'EntryDate' in trades_df.columns and 'EntryPrice' in trades_df.columns:
                        for _, trade in trades_df.iterrows():
                            if pd.notna(trade['EntryDate']) and pd.notna(trade['EntryPrice']):
                                ax1.plot(trade['EntryDate'], trade['EntryPrice'], 'g^', markersize=8)
                    
                    if 'ExitDate' in trades_df.columns and 'ExitPrice' in trades_df.columns:
                        for _, trade in trades_df.iterrows():
                            if pd.notna(trade['ExitDate']) and pd.notna(trade['ExitPrice']):
                                ax1.plot(trade['ExitDate'], trade['ExitPrice'], 'rv', markersize=8)
                except Exception as e:
                    logger.error(f"Error plotting trade markers: {str(e)}")
            
            # Calculate and plot drawdowns
            try:
                # First calculate the running maximum
                running_max = equity_curve['equity'].expanding().max()
                # Calculate drawdown in percentage terms
                drawdown = (equity_curve['equity'] / running_max - 1.0) * 100
                ax2.fill_between(equity_curve.index, 0, drawdown, color='red', alpha=0.3)
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_ylim(min(drawdown.min() * 1.1, -1), 1)  # Set some padding
            except Exception as e:
                logger.error(f"Error calculating drawdowns: {str(e)}")
            
            # Configure axis labels and title
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.set_title(f'Equity Curve - {symbol} ({exchange})')
            ax1.legend()
            ax1.grid(True)
            ax2.grid(True)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Save plot if requested
            if save_plots:
                fig.savefig(output_path / f"{symbol}_equity_curve.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error generating equity curve plot: {str(e)}")
        
        try:
            # 2. Return Distribution
            # Calculate returns from equity curve
            returns = equity_curve['equity'].pct_change().dropna()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot return distribution
            ax.hist(returns, bins=50, alpha=0.75, color='blue')
            
            # Add vertical line for mean return
            mean_return = returns.mean()
            ax.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_return:.4f}')
            
            # Add normal distribution fit
            import scipy.stats as stats
            x = np.linspace(returns.min(), returns.max(), 100)
            y = stats.norm.pdf(x, returns.mean(), returns.std())
            # Scale to match histogram height
            hist, bins = np.histogram(returns, bins=50)
            scale = max(hist) / max(y) if max(y) > 0 else 1
            ax.plot(x, y * scale, 'r-', linewidth=2, label='Normal Distribution Fit')
            
            ax.set_title(f'Return Distribution - {symbol} ({exchange})')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)
            
            if save_plots:
                fig.savefig(output_path / f"{symbol}_return_distribution.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error generating return distribution plot: {str(e)}")
        
        try:
            # 3. Rolling Metrics (Sharpe, Volatility)
            # Calculate window size for rolling metrics
            window = min(252, len(equity_curve) // 4)  # Use 1 year window or quarter of data
            if window > 20:  # Only calculate if we have enough data
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Calculate rolling metrics
                rolling_sharpe = returns.rolling(window=window).apply(lambda x: (x.mean() / x.std()) * np.sqrt(252), raw=True)
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                
                # Plot rolling Sharpe ratio
                ax1.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
                ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7)
                ax1.axhline(y=2, color='g', linestyle='--', alpha=0.7)
                ax1.set_ylabel('Rolling Sharpe Ratio')
                ax1.set_title(f'Rolling {window}-day Sharpe Ratio - {symbol} ({exchange})')
                
                # Plot rolling volatility
                ax2.plot(rolling_vol.index, rolling_vol * 100, linewidth=2, color='orange')  # Convert to percentage
                ax2.set_ylabel('Annualized Volatility (%)')
                ax2.set_title(f'Rolling {window}-day Volatility - {symbol} ({exchange})')
                
                # Format x-axis dates
                fig.autofmt_xdate()
                
                if save_plots:
                    fig.savefig(output_path / f"{symbol}_rolling_metrics.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error generating rolling metrics plot: {str(e)}")
        
        try:
            # 4. Trade Analysis
            if trades is not None and len(trades) > 0:
                # Convert to DataFrame if it's a list
                trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades
                
                # Check if we have PnL data
                if any(col in trades_df.columns for col in ['pnl', 'PnL']):
                    # Normalize column name
                    pnl_col = next((col for col in ['PnL', 'pnl'] if col in trades_df.columns), None)
                    
                    # Filter completed trades (those with exit information)
                    completed_trades = trades_df.dropna(subset=['ExitDate']) if 'ExitDate' in trades_df.columns else trades_df
                    
                    if len(completed_trades) > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot trade PnL as bars
                        bars = ax.bar(range(len(completed_trades)), completed_trades[pnl_col], 
                                    color=completed_trades[pnl_col].apply(lambda x: 'green' if x > 0 else 'red'))
                        
                        # Add horizontal line at zero
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        
                        # Label axes
                        ax.set_xlabel('Trade Number')
                        ax.set_ylabel('Profit/Loss ($)')
                        ax.set_title(f'Individual Trade Performance - {symbol} ({exchange})')
                        
                        # Add cumulative sum
                        if len(completed_trades) > 1:
                            ax2 = ax.twinx()
                            cumulative = completed_trades[pnl_col].cumsum()
                            ax2.plot(range(len(completed_trades)), cumulative, 'k--', linewidth=2)
                            ax2.set_ylabel('Cumulative P&L ($)')
                        
                        if save_plots:
                            fig.savefig(output_path / f"{symbol}_trade_pnl.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error generating trade analysis plot: {str(e)}")
        
        # Show plots if requested
        if show_plots:
            plt.show()
        else:
            plt.close('all')
    except Exception as e:
        logger.exception(f"Error in generate_plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
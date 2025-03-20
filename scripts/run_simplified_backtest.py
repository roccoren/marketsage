#!/usr/bin/env python
"""
Simplified backtest example without Qlib dependency
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.settings import OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_stock_list(file_path):
    """Read stock symbols from a file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def download_data(symbols, start_date, end_date):
    """Download historical data for the given symbols"""
    logger.info(f"Downloading data for {len(symbols)} symbols...")
    
    all_data = {}
    for symbol in symbols:
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                continue
                
            # Calculate technical indicators
            # Moving averages
            data['ma10'] = data['Close'].rolling(window=10).mean()
            data['ma20'] = data['Close'].rolling(window=20).mean()
            data['ma50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema12'] - data['ema26']
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            
            # Stochastic Oscillator
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['stoch_k'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
            
            # Bollinger Bands - fix for the DataFrame error
            bb_middle = data['Close'].rolling(window=20).mean()
            data['bb_middle'] = bb_middle
            std = data['Close'].rolling(window=20).std()
            data['bb_upper'] = bb_middle + (std * 2)
            data['bb_lower'] = bb_middle - (std * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            # Volume indicators
            data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # Momentum
            data['momentum'] = data['Close'].pct_change(periods=10) * 100
            
            # Trend
            data['trend_direction'] = np.where(data['ma10'] > data['ma20'], 1, -1)
            data['trend_strength'] = abs(data['ma10'] / data['ma20'] - 1) * 100
            
            # Clean up data (remove NaN values)
            data = data.dropna()
            
            # Store data
            all_data[symbol] = data
            logger.info(f"Downloaded and processed data for {symbol}: {len(data)} rows")
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
    
    return all_data

def generate_predictions(data, symbols, prediction_date):
    """Generate prediction scores for the given date"""
    predictions = {}
    
    if not data:  # Check if data is empty
        logger.warning("No data available for generating predictions")
        return pd.Series(predictions, dtype=float)  # Return empty Series with float type
    
    for symbol in symbols:
        if symbol not in data or data[symbol].empty:
            continue
        
        # Get data up to the prediction date
        df = data[symbol]
        df = df[df.index <= prediction_date]
        
        if df.empty:
            continue
        
        # Calculate prediction score based on technical indicators
        last_row = df.iloc[-1]
        
        rsi = float(last_row['rsi'])
        macd_hist = float(last_row['macd'] - last_row['signal'])
        stoch_k = float(last_row['stoch_k'])
        bb_width = float(last_row['bb_width'])
        volume_ratio = float(last_row['volume_ratio'])
        momentum = float(last_row['momentum'])
        trend_direction = float(last_row['trend_direction'])
        
        # RSI component (centered around 50)
        rsi_score = 1.0 - abs(rsi - 50) / 50
        
        # MACD component
        macd_score = np.tanh(macd_hist)
        
        # Stochastic component
        stoch_score = 1.0 if (stoch_k < 80 and stoch_k > 20) else 0.0
        
        # Bollinger width component (inverse - narrower is better)
        bb_score = 1.0 - min(bb_width, 0.1) / 0.1
        
        # Volume component
        volume_score = min(volume_ratio, 3.0) / 3.0
        
        # Momentum component
        mom_score = np.tanh(momentum / 5.0)
        
        # Trend component
        trend_score = 0.5 + (trend_direction * 0.5)
        
        # Combined score
        score = (
            rsi_score * 0.20 +
            macd_score * 0.25 +
            stoch_score * 0.10 +
            bb_score * 0.10 +
            volume_score * 0.10 +
            mom_score * 0.15 +
            trend_score * 0.10
        )
        
        predictions[symbol] = score
    
    return pd.Series(predictions, dtype=float)  # Specify float dtype to avoid issues with nlargest

def run_simplified_backtest(data, predictions, start_date, end_date, initial_capital=100000.0, topk=5):
    """
    Run a simplified backtest
    
    Parameters
    ----------
    data : dict
        Dictionary of DataFrames containing stock data
    predictions : pd.Series
        Series of prediction scores
    start_date : datetime
        Start date for backtesting
    end_date : datetime
        End date for backtesting
    initial_capital : float
        Initial capital
    topk : int
        Number of top stocks to hold
        
    Returns
    -------
    tuple
        (Portfolio DataFrame, Metrics Dict)
    """
    logger.info(f"Running simplified backtest from {start_date} to {end_date}")
    
    # Handle empty predictions
    if predictions.empty:
        logger.warning("No predictions available for backtesting")
        # Return empty portfolio and default metrics
        portfolio = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
        portfolio['cash'] = initial_capital
        portfolio['value'] = initial_capital
        portfolio['return'] = 0.0
        metrics = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
        }
        return portfolio, metrics
    
    # Get top stocks based on predictions
    top_stocks = predictions.nlargest(min(topk, len(predictions))).index.tolist()
    logger.info(f"Top {len(top_stocks)} stocks: {', '.join(top_stocks)}")
    
    # Initialize portfolio
    portfolio = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    portfolio['cash'] = initial_capital  # Start with all cash
    portfolio['value'] = initial_capital
    portfolio['return'] = 0.0
    
    # Add columns for each stock's position value
    for symbol in top_stocks:
        portfolio[symbol] = 0.0
    
    # Initialize positions (equal weight)
    position_value = initial_capital / len(top_stocks) if top_stocks else 0
    
    # For each stock, find the closest date in the data to start_date
    for symbol in top_stocks:
        if symbol not in data or data[symbol].empty:
            continue
            
        stock_data = data[symbol]
        
        # Find closest date on or after start_date
        valid_dates = stock_data.index[stock_data.index >= start_date]
        if valid_dates.empty:
            logger.warning(f"No data for {symbol} after {start_date}")
            continue
            
        entry_date = valid_dates[0]
        entry_price = stock_data.loc[entry_date, 'Close']
        shares = position_value / entry_price
        
        # Update portfolio
        portfolio.loc[entry_date:, 'cash'] -= position_value
        portfolio.loc[entry_date:, symbol] = shares * stock_data.loc[valid_dates, 'Close']
    
    # Calculate daily portfolio value
    for date in portfolio.index:
        portfolio.loc[date, 'value'] = portfolio.loc[date, 'cash'] + sum(
            portfolio.loc[date, symbol] for symbol in top_stocks if symbol in portfolio.columns
        )
    
    # Calculate daily returns
    portfolio['return'] = portfolio['value'].pct_change().fillna(0)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio)
    
    return portfolio, metrics

def calculate_metrics(portfolio):
    """Calculate performance metrics"""
    returns = portfolio['return'].values
    
    # Calculate metrics
    total_return = (portfolio['value'].iloc[-1] / portfolio['value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    daily_std = np.std(returns)
    annual_std = daily_std * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
    
    # Calculate drawdown
    drawdown = portfolio['value'] / portfolio['value'].cummax() - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    # Collect metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_std,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }
    
    return metrics

def plot_backtest_results(portfolio, benchmark_data=None, title="Backtest Results", save_path=None):
    """Plot backtest results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot portfolio value
    portfolio['value'].plot(ax=ax1, label='Portfolio Value')
    
    # Add benchmark if available
    if benchmark_data is not None and not benchmark_data.empty:
        # Normalize benchmark to start at the same value as portfolio
        benchmark_norm = benchmark_data['Close'] / benchmark_data['Close'].iloc[0] * portfolio['value'].iloc[0]
        benchmark_norm.plot(ax=ax1, label='Benchmark', alpha=0.7)
    
    ax1.set_title(title)
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot returns
    portfolio['return'].plot(ax=ax2, label='Daily Return')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    
    plt.close()

def save_backtest_results(portfolio, metrics, output_dir):
    """Save backtest results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save portfolio
    portfolio.to_csv(os.path.join(output_dir, 'portfolio.csv'))
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    # Create summary text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Backtest Results Summary\n")
        f.write("=======================\n\n")
        f.write(f"Total Return: {metrics['total_return']:.2%}\n")
        f.write(f"Annual Return: {metrics['annual_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
    
    logger.info(f"Saved results to {output_dir}")

def main():
    # Read stock list
    stock_file = 'config/stocks/backtest_sample.txt'
    symbols = read_stock_list(stock_file)
    logger.info(f"Read {len(symbols)} symbols from {stock_file}")
    
    # Define date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    backtest_start = end_date - timedelta(days=365)  # 1 year of backtest
    backtest_end = end_date
    
    # Download historical data
    data = download_data(symbols, start_date, end_date)
    
    # Download benchmark data (SPY)
    try:
        benchmark_data = yf.download('SPY', start=backtest_start, end=backtest_end, progress=False)
    except Exception as e:
        logger.error(f"Error downloading benchmark data: {str(e)}")
        benchmark_data = None
    
    # Generate predictions for the backtest start date
    predictions = generate_predictions(data, symbols, backtest_start)
    logger.info(f"Generated predictions for {len(predictions)} symbols at {backtest_start}")
    
    # Sort predictions and print the top and bottom 5
    if not predictions.empty:
        top_predictions = predictions.nlargest(min(5, len(predictions)))
        bottom_predictions = predictions.nsmallest(min(5, len(predictions)))
        
        logger.info("\nTop 5 predictions:")
        for symbol, score in top_predictions.items():
            logger.info(f"{symbol}: {score:.4f}")
        
        logger.info("\nBottom 5 predictions:")
        for symbol, score in bottom_predictions.items():
            logger.info(f"{symbol}: {score:.4f}")
    else:
        logger.warning("No predictions available to display")
    
    # Run backtest
    portfolio, metrics = run_simplified_backtest(
        data, 
        predictions, 
        backtest_start, 
        backtest_end,
        initial_capital=100000.0,
        topk=5
    )
    
    # Create output directory
    output_dir = OUTPUT_DIR / "backtest" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_backtest_results(portfolio, metrics, output_dir)
    
    # Plot results
    plot_path = os.path.join(output_dir, 'backtest_plot.png')
    plot_backtest_results(portfolio, benchmark_data, save_path=plot_path)
    
    # Print results
    logger.info("\nBacktest Results:")
    logger.info("=================")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    logger.info(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main() 
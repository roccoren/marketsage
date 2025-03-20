#!/usr/bin/env python
"""
Qlib Portfolio Strategy with Multiple Stocks
"""

import qlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Initialize Qlib
provider_uri = "data"
qlib.init(provider_uri=provider_uri)

def load_and_process_data(symbols):
    """Load and process data for multiple stocks"""
    all_data = {}
    for symbol in symbols:
        try:
            # Load data for the stock
            data_path = f"data/features/day/{symbol}.day.bin"
            stock_data = pd.read_pickle(data_path)
            
            if 'close' in stock_data.columns:
                # Add moving averages
                stock_data['ma10'] = stock_data['close'].rolling(window=10).mean()
                stock_data['ma20'] = stock_data['close'].rolling(window=20).mean()
                stock_data['ma50'] = stock_data['close'].rolling(window=50).mean()
                
                # Add RSI
                delta = stock_data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                stock_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Generate signals
                stock_data['signal'] = 0
                stock_data.loc[(stock_data['ma10'] > stock_data['ma20']) & (stock_data['rsi'] < 70), 'signal'] = 1
                stock_data.loc[(stock_data['ma10'] < stock_data['ma20']) | (stock_data['rsi'] > 70), 'signal'] = -1
                
                # Calculate returns
                stock_data['returns'] = stock_data['close'].pct_change()
                stock_data['strategy_returns'] = stock_data['signal'].shift(1) * stock_data['returns']
                
                # Drop NaN values
                stock_data = stock_data.dropna()
                
                # Store processed data
                all_data[symbol] = stock_data
            else:
                print(f"Cannot process {symbol}: 'close' column not found in data")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return all_data

def create_portfolio(all_data, rebalance_freq='M', equal_weight=True):
    """Create a portfolio of multiple stocks"""
    # Get common date range
    common_dates = None
    for symbol, data in all_data.items():
        if 'date' in data.columns:
            date_series = pd.to_datetime(data['date'])
            if common_dates is None:
                common_dates = set(date_series)
            else:
                common_dates = common_dates.intersection(set(date_series))
    
    if not common_dates:
        raise ValueError("No common dates found across stocks")
    
    common_dates = sorted(list(common_dates))
    start_date = min(common_dates)
    end_date = max(common_dates)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create portfolio dataframe
    portfolio = pd.DataFrame(index=date_range)
    portfolio.index.name = 'date'
    
    # Add strategy returns for each stock
    for symbol, data in all_data.items():
        # Convert string dates to datetime
        if isinstance(data['date'].iloc[0], str):
            data['date'] = pd.to_datetime(data['date'])
        
        # Set date as index for joining
        temp_data = data.set_index('date')
        
        # Add strategy returns to portfolio
        portfolio[f'{symbol}_returns'] = temp_data['returns']
        portfolio[f'{symbol}_strategy_returns'] = temp_data['strategy_returns']
        portfolio[f'{symbol}_signal'] = temp_data['signal']
    
    # Determine portfolio weights
    if equal_weight:
        n_stocks = len(all_data)
        weights = pd.DataFrame(index=portfolio.index)
        
        # Rebalance at specified frequency
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
        
        for symbol in all_data.keys():
            weights[symbol] = 0.0
        
        # Set initial weights
        current_weights = {symbol: 1.0/n_stocks for symbol in all_data.keys()}
        
        # Assign weights based on rebalancing dates
        for i in range(len(rebalance_dates)):
            start_idx = rebalance_dates[i]
            end_idx = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else end_date
            
            # Update weights based on signals at rebalance date
            active_symbols = []
            for symbol in all_data.keys():
                signal_col = f'{symbol}_signal'
                if signal_col in portfolio.columns:
                    signal = portfolio.loc[start_idx:start_idx, signal_col].values
                    if len(signal) > 0 and signal[0] == 1:
                        active_symbols.append(symbol)
            
            # Equal weight to active symbols, or stay in cash if none are active
            if active_symbols:
                for symbol in all_data.keys():
                    if symbol in active_symbols:
                        weights.loc[start_idx:end_idx, symbol] = 1.0 / len(active_symbols)
                    else:
                        weights.loc[start_idx:end_idx, symbol] = 0.0
        
        # Calculate portfolio returns
        portfolio['portfolio_returns'] = 0.0
        for symbol in all_data.keys():
            portfolio['portfolio_returns'] += portfolio[f'{symbol}_returns'] * weights[symbol]
        
        # Calculate cumulative returns
        portfolio['cum_returns'] = (1 + portfolio['portfolio_returns']).cumprod() - 1
        
        # Calculate strategy metrics
        total_return = portfolio['cum_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        daily_returns = portfolio['portfolio_returns']
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
        max_drawdown = (portfolio['cum_returns'] / portfolio['cum_returns'].cummax() - 1).min()
        
        return portfolio, {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'weights': weights
        }
    else:
        # For simplicity, we just use equal weights here
        # In a real system, you might implement a more sophisticated weighting scheme
        return create_portfolio(all_data, rebalance_freq, equal_weight=True)

def plot_portfolio_performance(portfolio, metrics, symbols):
    """Plot portfolio performance"""
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['cum_returns'], 'b-', linewidth=2)
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.savefig('outputs/portfolio_returns.png')
    
    # Plot individual stock signals
    plt.figure(figsize=(15, 10))
    for i, symbol in enumerate(symbols):
        signal_col = f'{symbol}_signal'
        if signal_col in portfolio.columns:
            plt.subplot(len(symbols), 1, i+1)
            plt.plot(portfolio.index, portfolio[signal_col], 'r-')
            plt.title(f'{symbol} Signal')
            plt.ylabel('Signal (-1/0/1)')
            plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/stock_signals.png')
    
    # Plot portfolio weights
    if 'weights' in metrics:
        weights = metrics['weights']
        plt.figure(figsize=(12, 6))
        for symbol in symbols:
            if symbol in weights.columns:
                plt.plot(weights.index, weights[symbol], linewidth=1, label=symbol)
        plt.title('Portfolio Weights Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/portfolio_weights.png')
    
    print(f"Portfolio performance charts saved to the 'outputs' directory")

def main():
    print("Starting portfolio strategy...")
    
    # Define symbols
    symbols = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
    
    # Load and process data
    print("Loading and processing data...")
    all_data = load_and_process_data(symbols)
    
    # Create portfolio
    print("Creating portfolio...")
    portfolio, metrics = create_portfolio(all_data, rebalance_freq='M')
    
    # Print portfolio metrics
    print("\nPortfolio Performance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Plot portfolio performance
    print("\nGenerating performance charts...")
    plot_portfolio_performance(portfolio, metrics, symbols)
    
    print("\nPortfolio strategy completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import logging

def generate_fake_data(num_stocks=50, start_date='2020-01-01', end_date='2020-12-31', freq='D'):
    """Generate fake stock data for testing"""
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create stock symbols
    stocks = [f"STOCK_{i:03d}" for i in range(1, num_stocks + 1)]
    
    # Create fake prediction dataframe
    data = {}
    for stock in stocks:
        # Random walk with positive drift for predictions
        preds = np.random.normal(0.001, 0.01, size=len(date_range)).cumsum()
        data[stock] = preds
    
    # Create dataframe with dates as index
    df = pd.DataFrame(data, index=date_range)
    
    return df


def run_backtest(pred_df, topk=10, n_drop=2, initial_capital=100000):
    """Run a simple backtest on the prediction dataframe"""
    print(f"Running backtest with {len(pred_df.columns)} stocks, topk={topk}, n_drop={n_drop}")
    
    # Portfolio value over time
    portfolio_values = pd.Series(index=pred_df.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital
    
    # Current positions
    positions = pd.DataFrame(0, index=pred_df.index, columns=pred_df.columns)
    
    # Track metrics
    daily_returns = []
    turnover_rates = []
    
    # Run backtest day by day
    for i, date in enumerate(pred_df.index):
        if i == 0:
            # Initial portfolio allocation
            top_stocks = pred_df.iloc[i].nlargest(topk).index
            positions.loc[date, top_stocks] = 1
            continue
        
        # Get top stocks for current date
        top_stocks = pred_df.iloc[i].nlargest(topk).index
        
        # Calculate turnover
        prev_positions = positions.iloc[i-1]
        new_positions = pd.Series(0, index=pred_df.columns)
        new_positions[top_stocks] = 1
        
        # Calculate turnover rate
        turnover = (new_positions != prev_positions).sum() / (2 * topk)
        turnover_rates.append(turnover)
        
        # Update positions
        positions.loc[date] = new_positions
        
        # Calculate daily return (random for demonstration)
        # In a real implementation, you would use actual price data
        daily_return = np.random.normal(0.0005, 0.01)  # Mean 0.05%, std 1%
        daily_returns.append(daily_return)
        
        # Update portfolio value
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return)
    
    # Calculate metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    annual_return = total_return / (len(pred_df) / 252)
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    max_drawdown = ((portfolio_values / portfolio_values.cummax()) - 1).min()
    
    # Print results
    print(f"Total Return: {total_return:.4f}")
    print(f"Annual Return: {annual_return:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Avg Turnover: {np.mean(turnover_rates):.4f}")
    
    # Plot portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values.index, portfolio_values.values)
    plt.title("Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    
    # Create results directory
    results_dir = os.path.join(os.getcwd(), "results", f"simple_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(results_dir, "portfolio_value.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Save results
    results = {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": np.mean(turnover_rates),
        "portfolio_values": portfolio_values
    }
    
    # Save as CSV and pickle
    results_df = pd.DataFrame({
        "metric": ["total_return", "annual_return", "sharpe", "max_drawdown", "turnover"],
        "value": [total_return, annual_return, sharpe, max_drawdown, np.mean(turnover_rates)]
    })
    results_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    
    with open(os.path.join(results_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_dir}")
    
    return results, results_dir


def main():
    """Run a simple test workflow"""
    parser = argparse.ArgumentParser(description="Simple test workflow")
    
    # Basic arguments
    parser.add_argument("--num_stocks", type=int, default=50, help="Number of stocks")
    parser.add_argument("--topk", type=int, default=10, help="Top-k stocks to select")
    parser.add_argument("--n_drop", type=int, default=2, help="Number of stocks to drop")
    
    args = parser.parse_args()
    
    print("Running simple test workflow...")
    
    try:
        # Generate fake prediction data
        print("Generating fake prediction data...")
        pred_df = generate_fake_data(
            num_stocks=args.num_stocks,
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Run backtest
        print("Running backtest...")
        results, results_dir = run_backtest(
            pred_df=pred_df,
            topk=args.topk,
            n_drop=args.n_drop
        )
        
        print(f"Test successful! Results saved to {results_dir}")
        return 0
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
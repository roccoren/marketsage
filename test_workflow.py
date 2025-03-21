#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the workflow implementation - consolidated testing framework
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import logging

# Add parent directory to path to import standalone_workflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_workflow import run_workflow


def generate_fake_data(num_stocks=50, start_date='2020-01-01', end_date='2020-12-31', freq='D'):
    """Generate fake stock data for testing"""
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create stock symbols
    stocks = [f"STOCK_{i:03d}" for i in range(1, num_stocks + 1)]
    
    # Create fake prediction dataframe
    data = []
    for date in date_range:
        for stock in stocks:
            # Generate some random prediction signals
            pred = np.random.randn()
            data.append({
                'datetime': date,
                'instrument': stock,
                'pred': pred,
            })
    
    # Convert to dataframe
    pred_df = pd.DataFrame(data)
    
    return pred_df


def run_backtest(pred_df, topk=10, n_drop=2, initial_capital=100000):
    """
    Simple backtester for testing purposes
    
    Parameters:
    -----------
    pred_df : pd.DataFrame
        Prediction dataframe with columns: datetime, instrument, pred
    topk : int
        Number of top stocks to select
    n_drop : int
        Number of stocks to drop from portfolio each day
    initial_capital : float
        Initial capital for backtesting
    
    Returns:
    --------
    pd.DataFrame
        Backtest results with daily portfolio value
    """
    # Ensure datetime is datetime type and sort
    pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
    pred_df = pred_df.sort_values('datetime')
    
    # Get unique dates
    dates = pred_df['datetime'].unique()
    
    # Initialize portfolio
    portfolio = {}
    portfolio_value = [initial_capital]
    capital_remaining = initial_capital
    portfolio_history = []
    
    # For each date
    for i, date in enumerate(dates):
        # Get predictions for current date
        day_pred = pred_df[pred_df['datetime'] == date]
        
        # Sort by prediction (descending)
        day_pred = day_pred.sort_values('pred', ascending=False)
        
        # Drop worst performing assets if we have any
        if len(portfolio) > 0 and n_drop > 0:
            # Sort current positions by prediction (ascending = worst first)
            current_positions = day_pred[day_pred['instrument'].isin(portfolio.keys())]
            current_positions = current_positions.sort_values('pred', ascending=True)
            
            # Get worst n_drop assets to sell
            to_sell = current_positions.head(min(n_drop, len(current_positions)))
            
            # Sell these assets
            for _, row in to_sell.iterrows():
                symbol = row['instrument']
                capital_remaining += portfolio.pop(symbol, 0)
        
        # Select top-k assets that we don't already own
        not_owned = day_pred[~day_pred['instrument'].isin(portfolio.keys())]
        to_buy = not_owned.head(topk - len(portfolio))
        
        # Allocate capital equally to new positions
        if len(to_buy) > 0:
            allocation_per_asset = capital_remaining / len(to_buy)
            for _, row in to_buy.iterrows():
                symbol = row['instrument']
                portfolio[symbol] = allocation_per_asset
                capital_remaining -= allocation_per_asset
        
        # Calculate portfolio value
        total_value = capital_remaining + sum(portfolio.values())
        portfolio_value.append(total_value)
        
        # Record portfolio state
        portfolio_history.append({
            'date': date,
            'portfolio_value': total_value,
            'capital_remaining': capital_remaining,
            'num_positions': len(portfolio)
        })
    
    # Create results dataframe
    results = pd.DataFrame(portfolio_history)
    
    return results


def plot_portfolio_value(results, save_path=None):
    """
    Plot portfolio value over time
    
    Parameters:
    -----------
    results : pd.DataFrame
        Backtest results
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results['date'], results['portfolio_value'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def run_simple_test(args):
    """Run a simplified test with fake data"""
    print(f"Running simple test with {args.num_stocks} stocks...")
    
    # Generate fake data
    pred_df = generate_fake_data(
        num_stocks=args.num_stocks,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    print(f"Generated {len(pred_df)} prediction records")
    
    # Run backtest
    results = run_backtest(
        pred_df,
        topk=args.topk,
        n_drop=args.n_drop,
        initial_capital=args.initial_capital
    )
    
    # Print summary
    initial_value = results['portfolio_value'].iloc[0]
    final_value = results['portfolio_value'].iloc[-1]
    returns = (final_value / initial_value - 1) * 100
    
    print(f"Backtest Results:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {returns:.2f}%")
    
    # Plot results
    if args.plot:
        plot_portfolio_value(results, save_path=args.output_file)
    
    return results


def main():
    """
    Run a simplified test of the stock analysis workflow
    """
    parser = argparse.ArgumentParser(description="Test stock analysis workflow")
    
    # Basic arguments
    parser.add_argument("--num_stocks", type=int, default=50, 
                        help="Number of stocks to simulate")
    parser.add_argument("--start_date", type=str, default="2020-01-01", 
                        help="Start date for simulation")
    parser.add_argument("--end_date", type=str, default="2020-12-31", 
                        help="End date for simulation")
    parser.add_argument("--topk", type=int, default=10, 
                        help="Number of top stocks to select")
    parser.add_argument("--n_drop", type=int, default=2, 
                        help="Number of stocks to drop from portfolio each day")
    parser.add_argument("--initial_capital", type=float, default=100000, 
                        help="Initial capital for backtesting")
    parser.add_argument("--plot", action="store_true", 
                        help="Plot the portfolio value")
    parser.add_argument("--output_file", type=str, default="portfolio_value.png", 
                        help="Output file for the plot")
    parser.add_argument("--use_real_workflow", action="store_true", 
                        help="Use the real workflow implementation instead of the simplified test")
    
    args = parser.parse_args()
    
    if args.use_real_workflow:
        # Convert args to the format expected by run_workflow
        workflow_args = argparse.Namespace(
            model="Linear",
            region="cn",
            market="csi300",
            start_date=args.start_date,
            end_date=args.end_date,
            topk=args.topk,
            n_drop=args.n_drop,
            skip_data=False,
            skip_prediction=False,
            skip_backtest=False,
            fine_tune=False,
            result_dir="./results/test",
            num_stocks=args.num_stocks,
            price_volatility=0.01,
            multi=False,
            train_start=args.start_date,
            train_end=None,  # Will be calculated automatically
            valid_start=None,
            valid_end=None,
            test_start=None,
            test_end=args.end_date,
            predict_start=None,
            predict_end=None,
            backtest_start=None,
            backtest_end=None,
            no_mlflow=True
        )
        
        print("Running test using real workflow implementation...")
        results = run_workflow(workflow_args)
        return 0
    else:
        print("Running simplified test implementation...")
        run_simple_test(args)
        return 0


if __name__ == "__main__":
    sys.exit(main()) 
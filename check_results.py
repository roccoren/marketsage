#!/usr/bin/env python3

import os
import sys
import pandas as pd
import pickle
import glob
import matplotlib.pyplot as plt

def main():
    """
    Check the date ranges in the most recent backtest results
    """
    # Find the most recent results directory
    results_dirs = glob.glob('results/stockscan_*')
    if not results_dirs:
        print("No results directories found")
        return 1
    
    latest_dir = max(results_dirs, key=os.path.getmtime)
    print(f"Checking results in {latest_dir}")
    
    # Check prediction file
    predictions_file = os.path.join(latest_dir, "model", "predictions.pkl")
    if os.path.exists(predictions_file):
        print("\nPredictions file:")
        with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)
        
        print(f"First date: {predictions.index[0]}")
        print(f"Last date: {predictions.index[-1]}")
        print(f"Date range length: {len(predictions)} days")
        print(f"Number of stocks: {len(predictions.columns)}")
    else:
        print(f"Predictions file not found at {predictions_file}")
    
    # Check backtest results from CSV file
    backtest_csv = os.path.join(latest_dir, "backtest", "backtest_results.csv")
    if os.path.exists(backtest_csv):
        print("\nBacktest results from CSV:")
        results_df = pd.read_csv(backtest_csv, index_col=0)
        print(results_df)
    
    # Check backtest portfolio values from PNG
    portfolio_png = os.path.join(latest_dir, "backtest", "portfolio_value.png")
    if os.path.exists(portfolio_png):
        print(f"\nPortfolio value chart found at {portfolio_png}")
    
    # Try to examine detailed results with different keys
    backtest_file = os.path.join(latest_dir, "backtest", "detailed_results.pkl")
    if os.path.exists(backtest_file):
        print("\nBacktest detailed results:")
        with open(backtest_file, 'rb') as f:
            results = pickle.load(f)
        
        # Print the keys in the results dictionary
        print(f"Keys in results dictionary: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
        
        # Try different possible keys for portfolio values
        portfolio_keys = ['portfolio_values', 'portfolio_value', 'portfolio', 'value', 'values']
        portfolio_found = False
        
        for key in portfolio_keys:
            if isinstance(results, dict) and key in results:
                portfolio_values = results[key]
                portfolio_found = True
                print(f"\nFound portfolio values under key '{key}':")
                print(f"First date: {portfolio_values.index[0]}")
                print(f"Last date: {portfolio_values.index[-1]}")
                print(f"Date range length: {len(portfolio_values)} days")
                break
        
        if not portfolio_found:
            print("Could not find portfolio values in detailed results.")
            
            # If results is a DataFrame, it might be the portfolio values directly
            if isinstance(results, pd.DataFrame) or isinstance(results, pd.Series):
                print("\nResults object is a DataFrame/Series, treating as portfolio values:")
                print(f"First date: {results.index[0]}")
                print(f"Last date: {results.index[-1]}")
                print(f"Date range length: {len(results)} days")
    else:
        print(f"Backtest file not found at {backtest_file}")
    
    # Check workflow log for parameters
    log_file = os.path.join(latest_dir, "workflow.log")
    if os.path.exists(log_file):
        print("\nParameters from workflow log:")
        search_terms = [
            "predict_start", "predict_end", 
            "backtest_start", "backtest_end",
            "price_volatility", "num_stocks"
        ]
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        # Look for the search terms in the log
        for term in search_terms:
            if term in log_content:
                # Extract the line with the term
                start_idx = log_content.find(term)
                if start_idx != -1:
                    # Find the start of the line
                    line_start = log_content.rfind('\n', 0, start_idx) + 1
                    # Find the end of the line
                    line_end = log_content.find('\n', start_idx)
                    line = log_content[line_start:line_end].strip()
                    print(f"  {line}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
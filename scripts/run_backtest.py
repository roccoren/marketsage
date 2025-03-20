#!/usr/bin/env python
import os
import sys
import logging
import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.qlib_backtest import QlibBacktest

def parse_args():
    parser = argparse.ArgumentParser(description='Run backtest')
    
    # Stock list file
    parser.add_argument('stock_list', type=str, help='Path to stock list file')
    
    # Time periods
    parser.add_argument('--historical-start', type=str, required=True, help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--historical-end', type=str, required=True, help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--backtest-start', type=str, required=True, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--backtest-end', type=str, required=True, help='End date for backtest (YYYY-MM-DD)')
    
    # Benchmark
    parser.add_argument('--benchmark', type=str, default='SPY', help='Benchmark symbol')
    
    # Strategy parameters
    parser.add_argument('--strategy', type=str, choices=['topk', 'enhanced_indexing'], default='topk', help='Strategy type')
    parser.add_argument('--topk', type=int, default=5, help='Number of stocks to select (for topk strategy)')
    parser.add_argument('--n-drop', type=int, default=1, help='Number of stocks to drop in each rebalance (for topk strategy)')
    
    # Account
    parser.add_argument('--account', type=float, default=100000.0, help='Initial account value')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/backtest', help='Output directory')

    # Analysis mode
    parser.add_argument('--analyze-symbol', type=str, help='Analyze a specific symbol for price targets')
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('outputs/backtest.log')
        ]
    )

def analyze_symbol(args):
    """Analyze a specific symbol for price targets and trading signals"""
    from src.data.loader import DataLoader
    from src.prediction.price_targets import analyze_stock_for_trading
    
    # Load data for the symbol
    loader = DataLoader()
    data = loader.load_data(
        symbols=[args.analyze_symbol],
        start_date=args.historical_start,
        end_date=args.backtest_end,
        include_technicals=True
    )
    
    if args.analyze_symbol not in data or data[args.analyze_symbol].empty:
        print(f"No data available for {args.analyze_symbol}")
        return
    
    # Analyze the symbol
    stock_data = data[args.analyze_symbol]
    analysis = analyze_stock_for_trading(stock_data)
    
    # Create output directory
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis to file
    with open(output_dir / f"{args.analyze_symbol}_analysis.json", "w") as f:
        json.dump(analysis, f, indent=4)
    
    # Print analysis
    print(f"\nAnalysis for {args.analyze_symbol} (as of {analysis['timestamp']}):")
    print(f"Current Price: ${analysis['current_price']}")
    print(f"Recommended Action: {analysis['action'].upper()}")
    print("\nPrice Targets:")
    print(f"Buy Price: ${analysis['price_targets']['buy_price']}")
    print(f"Sell Price: ${analysis['price_targets']['sell_price']}")
    print(f"Stop Loss: ${analysis['price_targets']['stop_loss']}")
    
    # Print prediction
    print("\nPrice Prediction:")
    print(f"Direction: {analysis['prediction']['direction']}")
    print(f"Forecast Price: ${analysis['prediction']['forecast_price']}")
    print(f"Confidence: {analysis['prediction']['confidence']}%")
    
    if analysis['risk_reward_ratio']:
        print(f"\nRisk/Reward Ratio: {analysis['risk_reward_ratio']}")
    
    print(f"\nSignal Strength - Buy: {analysis['buy_signal_strength']}, Sell: {analysis['sell_signal_strength']}")
    print(f"\nAnalysis saved to {output_dir}/{args.analyze_symbol}_analysis.json")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If analysis mode is enabled, analyze the symbol and exit
    if args.analyze_symbol:
        analyze_symbol(args)
        return
    
    # Check if stock list file exists
    if not os.path.exists(args.stock_list):
        logging.error(f"Stock list file not found: {args.stock_list}")
        return
    
    # Create strategy parameters
    strategy_params = {}
    if args.strategy == 'topk':
        strategy_params = {
            'topk': args.topk,
            'n_drop': args.n_drop
        }
    
    # Create and run backtest
    backtest = QlibBacktest(
        start_time=args.backtest_start,
        end_time=args.backtest_end,
        instruments=args.stock_list,
        benchmark=args.benchmark,
        account=args.account,
        strategy_type=args.strategy,
        strategy_params=strategy_params,
    )
    
    # Run backtest
    results = backtest.run_backtest(
        historical_start=args.historical_start,
        historical_end=args.historical_end
    )
    
    # Log results
    logging.info("Backtest Results:")
    logging.info(f"Total Return: {results['total_return']:.4f}")
    logging.info(f"Benchmark Return: {results['benchmark_return']:.4f}")
    logging.info(f"Annual Return: {results['annual_return']:.4f}")
    logging.info(f"Volatility: {results['volatility']:.4f}")
    logging.info(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    logging.info(f"Max Drawdown: {results['max_drawdown']:.4f}")
    logging.info(f"Turnover: {results['turnover']:.4f}")
    
    # Print price predictions for top performers
    if 'price_predictions' in results:
        print("\nPrice Predictions for Portfolio Stocks:")
        predictions = results['price_predictions']
        for symbol, data in predictions.items():
            print(f"\n{symbol}:")
            print(f"  Current Price: ${data['current_price']}")
            print(f"  Action: {data['action'].upper()}")
            print(f"  Buy Target: ${data['buy_price']}")
            print(f"  Sell Target: ${data['sell_price']}")
            print(f"  Stop Loss: ${data['stop_loss']}")
            print(f"  Forecast: ${data['forecast']} ({data['direction']}, {data['confidence']}% confidence)")
    
    # Print final message
    print(f"\nBacktest completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
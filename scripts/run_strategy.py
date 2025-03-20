#!/usr/bin/env python
"""
StockScan - Stock Market Analysis and Trading Strategy Runner
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.config.settings import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_SYMBOLS,
    DATA_DIR,
    OUTPUT_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_data(symbols, start_date, end_date, save_qlib=False):
    """Fetch and save stock data"""
    logger.info(f"Fetching data for symbols: {symbols}")
    loader = DataLoader(symbols, start_date, end_date)
    data = loader.load_data()
    
    if save_qlib:
        logger.info("Saving data in Qlib format...")
        loader.save_to_qlib_format()
    
    return data

def run_analysis(symbols, start_date, end_date):
    """Run performance analysis"""
    logger.info("Running performance analysis...")
    
    # Load data
    data = fetch_data(symbols, start_date, end_date)
    
    # Initialize strategy
    strategy = MomentumStrategy(symbols, start_date, end_date)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Calculate positions
    positions = strategy.calculate_positions(signals, data)
    
    # Calculate returns
    returns = strategy.calculate_returns(positions, data)
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(returns)
    
    # Generate visualizations
    strategy.plot_analysis(data, returns)
    
    # Print results
    logger.info("\nPerformance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def run_prediction(symbols, start_date, end_date):
    """Run prediction analysis"""
    logger.info("Running prediction analysis...")
    
    # Load data
    data = fetch_data(symbols, start_date, end_date)
    
    # Initialize strategy
    strategy = MomentumStrategy(symbols, start_date, end_date)
    
    # Generate predictions
    predictions = strategy.predict_signals(data)
    
    # Print latest predictions
    logger.info("\nLatest Predictions:")
    for symbol, pred in predictions.items():
        latest_pred = pred.iloc[-1]
        logger.info(f"{symbol}: {latest_pred:.2f}")
    
    return predictions

def run_strategy(symbols, start_date, end_date):
    """Run complete trading strategy"""
    logger.info("Running complete trading strategy...")
    
    # Load data
    data = fetch_data(symbols, start_date, end_date)
    
    # Initialize strategy
    strategy = MomentumStrategy(symbols, start_date, end_date)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Calculate positions
    positions = strategy.calculate_positions(signals, data)
    
    # Update current positions
    strategy.update_positions(positions, data)
    
    # Calculate returns
    returns = strategy.calculate_returns(positions, data)
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(returns)
    
    # Generate visualizations
    strategy.plot_analysis(data, returns)
    
    # Get strategy summary
    summary = strategy.get_strategy_summary()
    
    # Print results
    logger.info("\nStrategy Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("\nPerformance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return summary, metrics

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StockScan - Stock Market Analysis and Trading Strategy Runner")
    parser.add_argument("command", choices=["fetch", "analyze", "predict", "strategy"],
                      help="Command to execute")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                      help="Comma-separated list of stock symbols")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE,
                      help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE,
                      help="End date in YYYY-MM-DD format")
    parser.add_argument("--save-qlib", action="store_true",
                      help="Save data in Qlib format")
    
    args = parser.parse_args()
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Convert symbols string to list
    symbols = args.symbols.split(",")
    
    try:
        if args.command == "fetch":
            fetch_data(symbols, start_date, end_date, args.save_qlib)
        elif args.command == "analyze":
            run_analysis(symbols, start_date, end_date)
        elif args.command == "predict":
            run_prediction(symbols, start_date, end_date)
        elif args.command == "strategy":
            run_strategy(symbols, start_date, end_date)
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
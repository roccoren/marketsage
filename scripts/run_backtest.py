#!/usr/bin/env python
"""
Script to run backtesting with Qlib integration
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
from src.backtest.qlib_backtest import QlibBacktest
from src.config.settings import (
    DEFAULT_SYMBOLS,
    OUTPUT_DIR,
    PLOT_SETTINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_stock_list(file_path: str) -> list:
    """Read stock symbols from a file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def prepare_predictions(data: dict, symbols: list) -> pd.Series:
    """
    Prepare predictions from technical indicators
    
    Parameters
    ----------
    data : dict
        Dictionary of DataFrames containing stock data
    symbols : list
        List of stock symbols
        
    Returns
    -------
    pd.Series
        Series of prediction scores
    """
    predictions = {}
    
    for symbol in symbols:
        if symbol not in data or data[symbol].empty:
            continue
            
        df = data[symbol]
        
        # Calculate prediction score based on technical indicators
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd'].iloc[-1] - df['signal'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        momentum = df['momentum'].iloc[-1]
        trend_direction = df['trend_direction'].iloc[-1]
        trend_strength = df['trend_strength'].iloc[-1]
        
        # Combine indicators into a single score
        score = (
            (50 - abs(rsi - 50)) / 50 * 0.2 +  # RSI component (closer to 50 is better)
            np.tanh(macd_hist) * 0.2 +  # MACD component
            (50 - abs(stoch_k - 50)) / 50 * 0.1 +  # Stochastic component
            (1 - bb_width) * 0.1 +  # Volatility component (lower is better)
            (volume_ratio - 1) * 0.1 +  # Volume component
            momentum / 10 * 0.1 +  # Momentum component
            trend_direction * 0.1  # Trend direction component
        )
        
        predictions[symbol] = score
    
    return pd.Series(predictions)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run backtesting with Qlib integration")
    parser.add_argument("stock_file", help="File containing list of stock symbols")
    parser.add_argument("--historical-start", type=str, required=True,
                      help="Historical data start date in YYYY-MM-DD format")
    parser.add_argument("--historical-end", type=str, required=True,
                      help="Historical data end date in YYYY-MM-DD format")
    parser.add_argument("--backtest-start", type=str, required=True,
                      help="Backtest period start date in YYYY-MM-DD format")
    parser.add_argument("--backtest-end", type=str, required=True,
                      help="Backtest period end date in YYYY-MM-DD format")
    parser.add_argument("--benchmark", type=str, default="SPY",
                      help="Benchmark symbol for comparison")
    parser.add_argument("--strategy", type=str, default="topk",
                      choices=["topk", "enhanced_indexing"],
                      help="Strategy type to use")
    parser.add_argument("--topk", type=int, default=50,
                      help="Number of stocks to hold in portfolio")
    parser.add_argument("--n-drop", type=int, default=5,
                      help="Number of stocks to drop each rebalance")
    
    args = parser.parse_args()
    
    # Read stock list
    symbols = read_stock_list(args.stock_file)
    logger.info(f"Read {len(symbols)} symbols from {args.stock_file}")
    
    # Convert date strings to datetime objects
    historical_start = datetime.strptime(args.historical_start, "%Y-%m-%d")
    historical_end = datetime.strptime(args.historical_end, "%Y-%m-%d")
    backtest_start = datetime.strptime(args.backtest_start, "%Y-%m-%d")
    backtest_end = datetime.strptime(args.backtest_end, "%Y-%m-%d")
    
    # Create output directory
    backtest_output_dir = OUTPUT_DIR / "backtest"
    backtest_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load historical data
    logger.info("Loading historical data...")
    loader = DataLoader(symbols, historical_start, historical_end)
    data = loader.load_data()
    
    # Prepare predictions
    logger.info("Preparing predictions...")
    predictions = prepare_predictions(data, symbols)
    
    # Initialize backtesting system
    logger.info("Initializing backtesting system...")
    backtest_system = QlibBacktest(
        provider_uri=str(OUTPUT_DIR / "qlib_data"),
        region='us',
        start_time=args.backtest_start,
        end_time=args.backtest_end,
        instruments=symbols,
        benchmark=args.benchmark,
        strategy_type=args.strategy,
        strategy_params={
            'topk': args.topk,
            'n_drop': args.n_drop
        }
    )
    
    # Run backtest
    logger.info("Running backtest...")
    metrics, backtest_result = backtest_system.run_backtest(predictions)
    
    # Save results
    logger.info("Saving results...")
    backtest_system.save_results(backtest_result, metrics, str(backtest_output_dir))
    
    # Print results
    logger.info("\nBacktest Results:")
    logger.info("================")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Period: {args.backtest_start} to {args.backtest_end}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    logger.info(f"\nResults saved to: {backtest_output_dir}")

if __name__ == "__main__":
    main() 
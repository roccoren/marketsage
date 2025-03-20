#!/usr/bin/env python
"""
Run Qlib built-in strategy on a list of stock symbols from a file
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the parent directory to the path to import stockscan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockscan.core.qlib_manager import QlibManager
from stockscan.strategies.strategy_manager import StrategyManager
from stockscan.core.feature_manager import FeatureManager
from stockscan.models.model_manager import ModelManager
from stockscan.utils.env_setup import setup_environment, check_qlib_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qlib_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QlibStrategy')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Qlib built-in strategy on stock symbols')
    
    parser.add_argument('--symbols_file', type=str, required=True,
                        help='Path to the file containing stock symbols (one per line)')
    
    parser.add_argument('--provider_uri', type=str, default='~/.qlib/qlib_data/us_data',
                        help='URI for Qlib data provider')
    
    parser.add_argument('--region', type=str, default='us',
                        help='Region for Qlib data (e.g., us, cn)')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for analysis (format: YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for analysis (format: YYYY-MM-DD)')
    
    parser.add_argument('--strategy', type=str, default='topk',
                        choices=['topk', 'enhanced_indexing'],
                        help='Strategy type to use')
    
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top stocks to select (for topk strategy)')
    
    parser.add_argument('--n_drop', type=int, default=2,
                        help='Number of stocks to drop (for topk strategy)')
    
    parser.add_argument('--risk_degree', type=float, default=0.95,
                        help='Risk degree (for enhanced_indexing strategy)')
    
    parser.add_argument('--output_dir', type=str, default='./output/qlib_strategy',
                        help='Directory to save output files')
    
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot the backtest results')
    
    return parser.parse_args()

def read_symbols_file(file_path):
    """Read stock symbols from a file"""
    logger.info(f"Reading symbols from {file_path}")
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbols file: {e}")
        sys.exit(1)

def ensure_qlib_data(region):
    """Ensure Qlib data is available"""
    if not check_qlib_data(region):
        logger.error(f"Qlib data for region '{region}' not found.")
        logger.info("Please run 'python scripts/download_us_data.py' to download data.")
        sys.exit(1)

def save_results(results, output_path):
    """Save results to a file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert portfolio to CSV format
    if 'portfolio' in results and isinstance(results['portfolio'], pd.DataFrame):
        portfolio_path = os.path.join(os.path.dirname(output_path), 'portfolio.csv')
        results['portfolio'].to_csv(portfolio_path)
        logger.info(f"Portfolio saved to {portfolio_path}")
        
        # Replace portfolio with file path to avoid JSON serialization issues
        results['portfolio'] = portfolio_path
    
    # Save metrics to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {output_path}")

def plot_backtest_results(results, output_dir):
    """Plot backtest results"""
    try:
        import matplotlib.pyplot as plt
        
        # Plot portfolio value
        if 'portfolio' in results and isinstance(results['portfolio'], str):
            portfolio = pd.read_csv(results['portfolio'])
            portfolio['datetime'] = pd.to_datetime(portfolio['datetime'])
            portfolio.set_index('datetime', inplace=True)
            
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio['value'])
            plt.title('Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True)
            
            plot_path = os.path.join(output_dir, 'portfolio_value.png')
            plt.savefig(plot_path)
            logger.info(f"Portfolio value plot saved to {plot_path}")
            
            # Plot cumulative returns
            if 'return' in portfolio.columns:
                plt.figure(figsize=(12, 6))
                (1 + portfolio['return']).cumprod().plot()
                plt.title('Cumulative Returns')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return')
                plt.grid(True)
                
                returns_path = os.path.join(output_dir, 'cumulative_returns.png')
                plt.savefig(returns_path)
                logger.info(f"Cumulative returns plot saved to {returns_path}")
    except Exception as e:
        logger.warning(f"Error plotting results: {e}")

def main():
    """Main function"""
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Read symbols from file
    symbols = read_symbols_file(args.symbols_file)
    
    # Set up dates
    if args.start_date is None:
        # Default to 2 years ago
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        
    if args.end_date is None:
        # Default to today
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # Ensure we have Qlib data
    ensure_qlib_data(args.region)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Qlib
    logger.info("Initializing Qlib")
    qlib_manager = QlibManager(provider_uri=args.provider_uri, region=args.region)
    qlib_manager.init()
    
    # Initialize feature manager and load Alpha158 features
    logger.info("Loading Alpha158 features")
    feature_manager = FeatureManager()
    feature_manager.load_alpha158_features()
    fields = feature_manager.get_feature_list()
    
    # Create dataset handler
    logger.info(f"Creating dataset for {len(symbols)} symbols from {start_date} to {end_date}")
    handler = qlib_manager.create_dataset_handler(
        instruments=symbols,
        start_time=start_date,
        end_time=end_date,
        fields=fields
    )
    
    # Define segments
    train_cutoff = (datetime.strptime(start_date, '%Y-%m-%d') + 
                  (datetime.strptime(end_date, '%Y-%m-%d') - 
                   datetime.strptime(start_date, '%Y-%m-%d')) * 0.7).strftime('%Y-%m-%d')
    
    valid_cutoff = (datetime.strptime(start_date, '%Y-%m-%d') + 
                  (datetime.strptime(end_date, '%Y-%m-%d') - 
                   datetime.strptime(start_date, '%Y-%m-%d')) * 0.85).strftime('%Y-%m-%d')
    
    segments = {
        'train': (start_date, train_cutoff),
        'valid': (train_cutoff, valid_cutoff),
        'test': (valid_cutoff, end_date)
    }
    
    # Prepare dataset
    logger.info(f"Preparing dataset with segments: {segments}")
    dataset = qlib_manager.prepare_dataset(handler, segments)
    
    # Train a model
    logger.info("Training model")
    model_manager = ModelManager()
    model = model_manager.create_lgb_model("qlib_strategy_model")
    train_score, valid_score = model_manager.train_model(model.name, dataset)
    logger.info(f"Model trained - Train score: {train_score}, Valid score: {valid_score}")
    
    # Generate predictions
    logger.info("Generating predictions")
    predictions = model_manager.predict(model.name, dataset, segment='test')
    
    # Create a strategy
    logger.info(f"Creating {args.strategy} strategy")
    strategy_manager = StrategyManager()
    
    if args.strategy == 'topk':
        strategy = strategy_manager.create_topk_strategy(
            strategy_name="qlib_strategy",
            topk=args.topk,
            n_drop=args.n_drop
        )
    elif args.strategy == 'enhanced_indexing':
        strategy = strategy_manager.create_enhanced_indexing_strategy(
            strategy_name="qlib_strategy",
            market_view=predictions,
            risk_degree=args.risk_degree
        )
    
    # Run backtest
    logger.info("Running backtest")
    backtest_results = strategy_manager.backtest_strategy(
        strategy_name="qlib_strategy",
        pred_scores=predictions
    )
    
    # Evaluate strategy
    metrics = strategy_manager.evaluate_strategy(backtest_results)
    logger.info("Strategy evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'backtest_results.json')
    save_results({**backtest_results, 'metrics': metrics}, results_path)
    
    # Plot results if requested
    if args.plot_results:
        plot_backtest_results(backtest_results, args.output_dir)
    
    # Rank stocks based on predictions
    prediction_df = predictions.reset_index()
    max_date = prediction_df['datetime'].max()
    latest_preds = prediction_df[prediction_df['datetime'] == max_date]
    
    # Sort by prediction score (descending)
    ranked_stocks = latest_preds.sort_values(0, ascending=False)
    
    logger.info("\nStocks Ranked by Latest Prediction (Buy Signal Strength):")
    top_n = min(20, len(ranked_stocks))  # Show top 20 or less if fewer stocks
    for i, (index, row) in enumerate(ranked_stocks.head(top_n).iterrows(), 1):
        logger.info(f"{i}. {row['instrument']}: {row[0]:.6f}")
    
    # Save ranked stocks to CSV
    ranked_path = os.path.join(args.output_dir, 'ranked_stocks.csv')
    ranked_stocks.to_csv(ranked_path)
    logger.info(f"Ranked stocks saved to {ranked_path}")
    
    logger.info(f"Complete - All results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
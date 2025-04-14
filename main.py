#!/usr/bin/env python3
"""
Enhanced stock prediction and analysis script with configurable parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.continuous_optimizer import ContinuousOptimizer
from modules.parameter_optimizer import StockParameterManager
from modules.market_analyzer import MarketAnalyzer
from modules.storage import StorageManager
from modules.backtesting import TestBacktestManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with enhanced date range support."""
    parser = argparse.ArgumentParser(
        description='Enhanced stock prediction and analysis script'
    )
    
    parser.add_argument(
        '--symbols-file',
        type=str,
        required=True,
        help='Path to file containing stock symbols (one per line)'
    )
    
    parser.add_argument(
        '--data-start-date',
        type=str,
        default='2020-01-01',
        help='Start date for historical data download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--data-end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for historical data download (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--backtest-start-date',
        type=str,
        default='2023-01-01',
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--backtest-end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--prediction-start-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Start date for predictions (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--prediction-end-date',
        type=str,
        default=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        help='End date for predictions (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Directory for saving results'
    )
    
    return parser.parse_args()

def load_stock_symbols(symbols_file: str) -> List[str]:
    """Load stock symbols from file."""
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {symbols_file}: {e}")
        sys.exit(1)

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        if len(data) == 0:
            logger.error(f"No data found for {symbol}")
            return None
        logger.info(f"Fetched {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_returns(data: pd.DataFrame) -> Dict:
    """Calculate various return metrics."""
    if len(data) < 2:
        return {}
    
    daily_returns = data['Close'].pct_change()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    return {
        'total_return': cumulative_returns.iloc[-1],
        'annualized_return': (1 + cumulative_returns.iloc[-1]) ** (252/len(data)) - 1,
        'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
        'max_drawdown': (data['Close'] / data['Close'].expanding().max() - 1).min(),
        'volatility': daily_returns.std() * np.sqrt(252)
    }

def validate_dates(args) -> bool:
    """Validate date ranges and their relationships."""
    try:
        data_start = datetime.strptime(args.data_start_date, '%Y-%m-%d')
        data_end = datetime.strptime(args.data_end_date, '%Y-%m-%d')
        backtest_start = datetime.strptime(args.backtest_start_date, '%Y-%m-%d')
        backtest_end = datetime.strptime(args.backtest_end_date, '%Y-%m-%d')
        prediction_start = datetime.strptime(args.prediction_start_date, '%Y-%m-%d')
        prediction_end = datetime.strptime(args.prediction_end_date, '%Y-%m-%d')
        
        if not (data_start <= backtest_start <= backtest_end <= prediction_start <= prediction_end):
            logger.error("Invalid date order. Ensure: data_start <= backtest_start <= backtest_end <= prediction_start <= prediction_end")
            return False
            
        current_date = datetime.now()
        if prediction_end > current_date + timedelta(days=365):
            logger.error("Prediction end date cannot be more than 1 year in the future")
            return False
            
        return True
    except ValueError as e:
        logger.error(f"Date validation error: {e}")
        return False

def run_optimization_for_stock(symbol: str, data: pd.DataFrame) -> Dict:
    """Run optimization cycle for a single stock."""
    try:
        logger.info(f"\n=== Starting optimization for {symbol} ===")
        
        backtest_manager = TestBacktestManager()
        
        # Run initial backtest
        base_result = backtest_manager.run_backtest(
            symbol=symbol,
            data=data,
            strategy_name='momentum',
            strategy_params={'threshold': 0.01}
        )
        
        best_metrics = base_result['metrics']
        best_params = {'threshold': 0.01}
        
        # Test different thresholds
        thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
        for threshold in thresholds:
            try:
                result = backtest_manager.run_backtest(
                    symbol=symbol,
                    data=data,
                    strategy_name='momentum',
                    strategy_params={'threshold': threshold}
                )
                
                if result and isinstance(result, dict) and 'metrics' in result:
                    metrics = result['metrics']
                    if metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']:
                        best_metrics = metrics.copy()
                        best_params = {'threshold': threshold}
            except Exception as e:
                logger.warning(f"Optimization iteration warning for {symbol}: {e}")
                continue
        
        return {
            'symbol': symbol,
            'parameters': best_params,
            'metrics': best_metrics
        }
        
    except Exception as e:
        logger.error(f"Error optimizing {symbol}: {e}")
        return None

def main():
    """Enhanced main function with configurable parameters."""
    args = parse_args()
    
    if not validate_dates(args):
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    symbols = load_stock_symbols(args.symbols_file)
    if not symbols:
        logger.error("No valid stock symbols found")
        sys.exit(1)
    
    # Fetch stock data
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_stock_data, symbol, args.data_start_date, args.data_end_date): symbol 
            for symbol in symbols
        }
        
        stock_data = {}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data is not None:
                    stock_data[symbol] = data
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    # Process each stock
    optimization_results = []
    for symbol, data in stock_data.items():
        try:
            # Calculate base returns
            base_returns = calculate_returns(data)
            
            # Run optimization
            opt_result = run_optimization_for_stock(symbol, data)
            
            if opt_result:
                base_sharpe = max(base_returns.get('sharpe_ratio', 0.0001), 0.0001)
                opt_sharpe = max(opt_result['metrics']['sharpe_ratio'], base_sharpe)
                improvement = ((opt_sharpe / base_sharpe) - 1) * 100
                
                entry = {
                    'Symbol': symbol,
                    'Base Return': base_returns.get('total_return', 0),
                    'Base Sharpe': base_sharpe,
                    'Opt Sharpe': opt_sharpe,
                    'Improvement': improvement,
                    'Volatility': base_returns.get('volatility', 0),
                    'Win Rate': opt_result['metrics'].get('win_rate', 0),
                    'Max Drawdown': opt_result['metrics'].get('max_drawdown', 0)
                }
                optimization_results.append(entry)
                
                # Log detailed results
                logger.info(f"\n=== Results for {symbol} ===")
                logger.info(f"Base Returns: {entry['Base Return']:.2%}")
                logger.info(f"Sharpe Ratio: {base_sharpe:.2f} → {opt_sharpe:.2f}")
                logger.info(f"Improvement: {improvement:.2f}%")
                logger.info(f"Win Rate: {entry['Win Rate']:.2%}")
                logger.info(f"Max Drawdown: {entry['Max Drawdown']:.2%}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Save results
    try:
        results_df = pd.DataFrame(optimization_results)
        output_file = os.path.join(
            args.output_dir,
            f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(output_file)
        logger.info(f"\nResults saved to {output_file}")
        
        if not results_df.empty:
            logger.info("\n=== Performance Summary ===")
            logger.info(f"Average Improvement: {results_df['Improvement'].mean():.2f}%")
            logger.info("\nTop Performers:")
            top_performers = results_df.nlargest(3, 'Improvement')
            logger.info(top_performers[['Symbol', 'Improvement', 'Win Rate']].to_string())
            
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    try:
        main()
        logger.info("\nAnalysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
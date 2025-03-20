#!/usr/bin/env python
"""
Batch processing script for running strategies on multiple stocks
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict
import json
import traceback

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.config.settings import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DATA_DIR,
    OUTPUT_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_stock_list(file_path: str) -> List[str]:
    """Read stock symbols from a file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_stock_batch(symbols: List[str], start_date: datetime, end_date: datetime, batch_size: int = 20):
    """Process stocks in batches"""
    results = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} stocks")
        
        try:
            # Load data for the batch
            loader = DataLoader(batch, start_date, end_date)
            data = loader.load_data()
            
            # Filter out symbols with no data
            valid_symbols = [symbol for symbol in batch if symbol in data and not data[symbol].empty]
            if not valid_symbols:
                logger.warning(f"No valid data found for batch {i//batch_size + 1}")
                continue
                
            logger.info(f"Found valid data for {len(valid_symbols)} symbols out of {len(batch)}")
            
            # Initialize and run strategy with only valid symbols
            strategy = MomentumStrategy(valid_symbols, start_date, end_date)
            signals = strategy.generate_signals(data)
            
            # Ensure all DataFrames have the same index before calculating positions
            common_index = None
            for symbol in valid_symbols:
                if symbol in data and not data[symbol].empty:
                    if common_index is None:
                        common_index = data[symbol].index
                    else:
                        common_index = common_index.intersection(data[symbol].index)
            
            # Reindex all data to the common index
            if common_index is not None and len(common_index) > 0:
                aligned_data = {}
                for symbol in valid_symbols:
                    if symbol in data and not data[symbol].empty:
                        aligned_data[symbol] = data[symbol].reindex(common_index)
                
                # Recalculate signals with aligned data
                signals = strategy.generate_signals(aligned_data)
                positions = strategy.calculate_positions(signals, aligned_data)
                returns = strategy.calculate_returns(positions, aligned_data)
                metrics = strategy.calculate_metrics(returns)
                
                # Store results
                results.append({
                    'batch': i//batch_size + 1,
                    'symbols': valid_symbols,
                    'metrics': metrics,
                    'returns': returns.tolist(),
                    'dates': returns.index.strftime('%Y-%m-%d').tolist()
                })
                
                # Generate visualizations for the batch
                try:
                    strategy.plot_analysis(aligned_data, returns)
                except Exception as e:
                    logger.error(f"Error generating visualizations for batch {i//batch_size + 1}: {str(e)}")
            else:
                logger.warning(f"No common index found for batch {i//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    return results

def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results from all batches"""
    if not results:
        return {'aggregate_metrics': {}, 'batch_results': []}
    
    try:
        # Combine returns from all batches
        all_returns = pd.Series(dtype=float)
        for batch_result in results:
            try:
                if 'returns' in batch_result and 'dates' in batch_result:
                    batch_returns = pd.Series(
                        batch_result['returns'],
                        index=pd.to_datetime(batch_result['dates'])
                    )
                    
                    # Ensure index has no duplicates
                    if batch_returns.index.duplicated().any():
                        logger.warning(f"Duplicate dates found in batch {batch_result['batch']}, removing duplicates")
                        batch_returns = batch_returns[~batch_returns.index.duplicated()]
                    
                    if all_returns.empty:
                        all_returns = batch_returns
                    else:
                        # Ensure indices are aligned for concatenation
                        combined_index = all_returns.index.union(batch_returns.index)
                        all_returns = all_returns.reindex(combined_index, fill_value=0.0)
                        batch_returns = batch_returns.reindex(combined_index, fill_value=0.0)
                        
                        # Average the returns
                        all_returns = (all_returns + batch_returns) / 2
            except Exception as e:
                logger.error(f"Error aggregating returns for batch {batch_result.get('batch', 'unknown')}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        if all_returns.empty:
            logger.warning("No valid returns to calculate aggregate metrics")
            aggregate_metrics = {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        else:
            try:
                strategy = MomentumStrategy([], None, None)  # Dummy strategy for metrics calculation
                aggregate_metrics = strategy.calculate_metrics(all_returns)
            except Exception as e:
                logger.error(f"Error calculating aggregate metrics: {str(e)}")
                aggregate_metrics = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0
                }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'batch_results': results
        }
    except Exception as e:
        logger.error(f"Error in aggregate_results: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'aggregate_metrics': {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            },
            'batch_results': results
        }

def save_results(results: Dict, output_dir: Path):
    """Save results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Ensure metrics are serializable
        if 'aggregate_metrics' in results:
            serializable_metrics = {}
            for k, v in results['aggregate_metrics'].items():
                if isinstance(v, (np.float64, np.float32)):
                    serializable_metrics[k] = float(v)
                else:
                    serializable_metrics[k] = v
        else:
            serializable_metrics = {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
            
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        # Prepare batch results for serialization
        serializable_batches = []
        for batch in results.get('batch_results', []):
            serializable_batch = {}
            for k, v in batch.items():
                if k == 'metrics':
                    serializable_batch[k] = {
                        mk: float(mv) if isinstance(mv, (np.float64, np.float32)) else mv
                        for mk, mv in v.items()
                    }
                else:
                    serializable_batch[k] = v
            serializable_batches.append(serializable_batch)
            
        # Save batch results
        with open(output_dir / 'batch_results.json', 'w') as f:
            json.dump(serializable_batches, f, indent=4)
        
        # Save summary report
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("Strategy Performance Summary\n")
            f.write("==========================\n\n")
            
            f.write("Aggregate Metrics:\n")
            for metric, value in serializable_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nBatch Results:\n")
            for batch in serializable_batches:
                f.write(f"\nBatch {batch['batch']} ({len(batch['symbols'])} stocks):\n")
                for metric, value in batch['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Batch processing script for running strategies on multiple stocks")
    parser.add_argument("stock_file", help="File containing list of stock symbols")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE,
                      help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE,
                      help="End date in YYYY-MM-DD format")
    parser.add_argument("--batch-size", type=int, default=5,
                      help="Number of stocks to process in each batch")
    
    args = parser.parse_args()
    
    # Read stock list
    symbols = read_stock_list(args.stock_file)
    logger.info(f"Read {len(symbols)} symbols from {args.stock_file}")
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Process stocks in batches
    results = process_stock_batch(symbols, start_date, end_date, args.batch_size)
    
    # Aggregate results
    aggregated_results = aggregate_results(results)
    
    # Save results
    save_results(aggregated_results, OUTPUT_DIR / "batch_results")
    
    # Print summary
    logger.info("\nStrategy Performance Summary:")
    for metric, value in aggregated_results.get('aggregate_metrics', {}).items():
        if isinstance(value, (float, int)):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main() 
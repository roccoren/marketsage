#!/usr/bin/env python3
"""
Script to optimize market data files for performance.
Uses the optimized processor and normalizer classes.
"""

import argparse
import logging
from pathlib import Path
import sys
import concurrent.futures
from tqdm import tqdm

from .core.utils import ProcessingStats, log_system_stats
from .optimized_market_processor import OptimizedMarketProcessor, benchmark_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_market_directory(directory: str, batch_size: int = 1000, max_workers: int = None) -> None:
    """
    Process and optimize market data files in a directory.
    
    Args:
        directory: Directory containing market data
        batch_size: Files to process in each batch
        max_workers: Maximum number of worker processes
    """
    data_path = Path(directory)
    if not data_path.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    # Create processor
    processor = OptimizedMarketProcessor(data_path)
    
    # Get markets to process
    markets = [d.name for d in data_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(markets)} markets to process")
    
    # Process each market
    for market in markets:
        logger.info(f"Processing market: {market}")
        
        try:
            # Process and optimize market data
            df = processor.process_market_data(market, parallel=True)
            
            # Log results
            logger.info(f"Successfully processed market {market}")
            logger.info(f"Data shape: {df.shape}")
            
            # Log performance metrics
            metrics = processor.get_performance_metrics()
            logger.info(f"Performance metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error processing market {market}: {e}")
    
    logger.info("Market data optimization complete")

def main():
    """Main function for the optimize_market_data script."""
    parser = argparse.ArgumentParser(description="Optimize market data files for performance.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to market data directory")
    parser.add_argument("--market", type=str, help="Specific market to process")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark tests comparing parallel vs sequential")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of files to process in each batch")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (defaults to CPU count - 1)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    if args.benchmark:
        # Run benchmarks
        if args.market:
            logger.info(f"Running benchmark for market: {args.market}")
            results = benchmark_processor(data_dir, args.market)
            
            logger.info("Benchmark results:")
            logger.info(f"Parallel processing: {results['parallel']['time_seconds']} seconds")
            logger.info(f"Sequential processing: {results['sequential']['time_seconds']} seconds")
            logger.info(f"Speedup factor: {results['speedup_factor']}x")
        else:
            logger.error("Must specify a market for benchmarking with --market")
            return 1
    elif args.market:
        # Process specific market
        logger.info(f"Processing market: {args.market}")
        
        processor = OptimizedMarketProcessor(data_dir)
        
        try:
            df = processor.process_market_data(args.market, parallel=True)
            
            logger.info(f"Successfully processed market {args.market}")
            logger.info(f"Data shape: {df.shape}")
            
            metrics = processor.get_performance_metrics()
            logger.info(f"Performance metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error processing market {args.market}: {e}")
            return 1
    else:
        # Process all markets
        process_market_directory(
            data_dir, 
            batch_size=args.batch_size, 
            max_workers=args.workers
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
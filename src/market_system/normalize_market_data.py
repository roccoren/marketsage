#!/usr/bin/env python3
"""
Script to normalize market data files.
Uses the base MarketDataNormalizer class.
"""

import argparse
import logging
from pathlib import Path
import sys

from .core.normalizer import MarketDataNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Process market data files for normalization."""
    parser = argparse.ArgumentParser(description="Normalize market data files.")
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Path to market data directory")
    parser.add_argument("--market", type=str, help="Specific market to process")
    parser.add_argument("--symbol", type=str, help="Specific symbol to process")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    normalizer = MarketDataNormalizer(data_dir)
    
    if args.market and args.symbol:
        # Process specific symbol
        market_dir = data_dir / args.market
        symbol_dir = market_dir / args.symbol
        
        if not symbol_dir.exists():
            logger.error(f"Symbol directory does not exist: {symbol_dir}")
            return 1
            
        logger.info(f"Processing symbol {args.symbol} in market {args.market}")
        
        files_processed = 0
        files_failed = 0
        
        for file_path in symbol_dir.glob('*.json'):
            if normalizer.process_file(file_path):
                files_processed += 1
            else:
                files_failed += 1
                
        logger.info(f"Processed {files_processed} files, failed {files_failed} files")
        
    elif args.market:
        # Process specific market
        market_dir = data_dir / args.market
        
        if not market_dir.exists():
            logger.error(f"Market directory does not exist: {market_dir}")
            return 1
            
        logger.info(f"Processing market {args.market}")
        
        for symbol_dir in market_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
                
            symbol = symbol_dir.name
            logger.info(f"Processing symbol {symbol}")
            
            files_processed = 0
            files_failed = 0
            
            for file_path in symbol_dir.glob('*.json'):
                if normalizer.process_file(file_path):
                    files_processed += 1
                else:
                    files_failed += 1
                    
            logger.info(f"Symbol {symbol}: Processed {files_processed} files, failed {files_failed} files")
            
    else:
        # Process all markets
        logger.info("Processing all markets")
        stats = normalizer.process_all_files()
        
        logger.info(f"Total processed: {stats['processed']}")
        logger.info(f"Total failed: {stats['failed']}")
        logger.info(f"Total files: {stats['total']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Simplified script to download stock data using Qlib
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to Python path to import local modules
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from stockscan.core.qlib_simple import QlibSimpleInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_symbols_from_file(file_path: str) -> List[str]:
    """Load symbols from a text file."""
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {file_path}: {str(e)}")
        return []

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download stock data using Qlib'
    )
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--symbols-file', type=str, help='File containing symbols (one per line)')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--region', type=str, default='us', choices=['us', 'cn'], help='Region code')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory for data')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load symbols
        symbols = []
        if args.symbols_file:
            symbols = load_symbols_from_file(args.symbols_file)
        elif args.symbols:
            symbols = args.symbols.split(',')
        
        if not symbols:
            logger.error("No symbols specified. Use --symbols or --symbols-file")
            return
        
        # Initialize Qlib interface
        qlib_interface = QlibSimpleInterface(region=args.region)
        
        # Download data
        output_dir = args.output_dir
        if not output_dir:
            output_dir = os.path.join(project_root, "data")
            
        result = qlib_interface.download_data(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            target_dir=output_dir
        )
        
        logger.info(f"Downloaded data for {len(result)} symbols to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
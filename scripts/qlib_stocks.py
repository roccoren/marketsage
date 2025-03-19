#!/usr/bin/env python3
"""
Qlib-based command-line utility for stock data operations and strategy execution
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

from stockscan.core.qlib_interface import QlibInterface

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

def command_download_data(args):
    """Command to download stock data."""
    logger.info("Starting data download")
    
    # Load symbols from file if specified
    symbols = []
    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif args.symbols:
        symbols = args.symbols.split(',')
    
    if not symbols:
        logger.error("No symbols specified. Use --symbols or --symbols-file")
        return
    
    # Initialize Qlib interface
    qlib_interface = QlibInterface(region=args.region)
    
    # Download data
    target_dir = args.output_dir if args.output_dir else os.path.join(project_root, "data")
    result = qlib_interface.download_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        target_dir=target_dir
    )
    
    logger.info(f"Downloaded data for {len(result)} symbols to {target_dir}")

def command_run_strategy(args):
    """Command to run a trading strategy."""
    logger.info(f"Running strategy: {args.strategy}")
    
    # Load symbols from file if specified
    symbols = []
    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif args.symbols:
        symbols = args.symbols.split(',')
    
    if not symbols:
        logger.error("No symbols specified. Use --symbols or --symbols-file")
        return
    
    # Load strategy parameters if specified
    strategy_params = {}
    if args.params_file:
        import json
        try:
            with open(args.params_file, 'r') as f:
                strategy_params = json.load(f)
        except Exception as e:
            logger.error(f"Error loading strategy parameters: {str(e)}")
    
    # Initialize Qlib interface
    qlib_interface = QlibInterface(region=args.region)
    
    # Initialize Qlib
    data_path = os.path.join(project_root, "data") if not args.data_path else args.data_path
    qlib_interface.init_qlib(data_path)
    
    # Run strategy
    result = qlib_interface.run_strategy(
        strategy_name=args.strategy,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_params=strategy_params
    )
    
    # Save results
    if args.output_file:
        import json
        try:
            # Convert non-serializable objects
            import pandas as pd
            for key, value in result.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    result[key] = value.to_dict()
            
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    # Display summary
    logger.info("Strategy Run Summary:")
    logger.info(f"Strategy: {result.get('strategy')}")
    logger.info(f"Period: {result.get('period')}")
    
    # Display key metrics if available
    metrics = result.get('metrics', {})
    if metrics:
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, (dict, list)):
                logger.info(f"  {key}: {value}")

def command_scan_market(args):
    """Command to scan market for trading signals."""
    logger.info(f"Scanning market using strategy: {args.strategy}")
    
    # Load symbols from file if specified
    symbols = []
    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif args.symbols:
        symbols = args.symbols.split(',')
    
    if not symbols:
        logger.error("No symbols specified. Use --symbols or --symbols-file")
        return
    
    # Load strategy parameters if specified
    strategy_params = {}
    if args.params_file:
        import json
        try:
            with open(args.params_file, 'r') as f:
                strategy_params = json.load(f)
        except Exception as e:
            logger.error(f"Error loading strategy parameters: {str(e)}")
    
    # Initialize Qlib interface
    qlib_interface = QlibInterface(region=args.region)
    
    # Initialize Qlib
    data_path = os.path.join(project_root, "data") if not args.data_path else args.data_path
    qlib_interface.init_qlib(data_path)
    
    # Scan market
    result = qlib_interface.scan_market(
        strategy_name=args.strategy,
        symbols=symbols,
        date=args.date,
        top_k=args.top_k,
        strategy_params=strategy_params
    )
    
    # Save results
    if args.output_file:
        import json
        try:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    # Display summary
    logger.info(f"Market Scan Results for {result.get('strategy')} on {result.get('date')}:")
    
    # Display buy signals
    buy_signals = result.get('buy_signals', [])
    if buy_signals:
        logger.info("Top Buy Signals:")
        for i, signal in enumerate(buy_signals, 1):
            logger.info(f"  {i}. {signal.get('symbol')}: {signal.get('score'):.4f}")
    
    # Display sell signals
    sell_signals = result.get('sell_signals', [])
    if sell_signals:
        logger.info("Top Sell Signals:")
        for i, signal in enumerate(sell_signals, 1):
            logger.info(f"  {i}. {signal.get('symbol')}: {signal.get('score'):.4f}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Qlib-based stock data and strategy utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download data:     python qlib_stocks.py download --symbols-file config/market/test-symbols.txt --start-date 2022-01-01
  Run strategy:      python qlib_stocks.py run-strategy --strategy gbdt --symbols AAPL,MSFT --start-date 2022-01-01
  Scan market:       python qlib_stocks.py scan-market --strategy gbdt --symbols-file config/market/test-symbols.txt
"""
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Download data command
    download_parser = subparsers.add_parser('download', help='Download stock data')
    download_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    download_parser.add_argument('--symbols-file', type=str, help='File containing symbols (one per line)')
    download_parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), defaults to today')
    download_parser.add_argument('--region', type=str, default='us', choices=['us', 'cn'], help='Region code')
    download_parser.add_argument('--output-dir', type=str, help='Output directory for data')
    
    # Run strategy command
    strategy_parser = subparsers.add_parser('run-strategy', help='Run a trading strategy')
    strategy_parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    strategy_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    strategy_parser.add_argument('--symbols-file', type=str, help='File containing symbols (one per line)')
    strategy_parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    strategy_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), defaults to today')
    strategy_parser.add_argument('--region', type=str, default='us', choices=['us', 'cn'], help='Region code')
    strategy_parser.add_argument('--params-file', type=str, help='JSON file with strategy parameters')
    strategy_parser.add_argument('--data-path', type=str, help='Path to Qlib data directory')
    strategy_parser.add_argument('--output-file', type=str, help='Output file for results')
    
    # Scan market command
    scan_parser = subparsers.add_parser('scan-market', help='Scan market for trading signals')
    scan_parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    scan_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    scan_parser.add_argument('--symbols-file', type=str, help='File containing symbols (one per line)')
    scan_parser.add_argument('--date', type=str, help='Date to scan (YYYY-MM-DD), defaults to today')
    scan_parser.add_argument('--region', type=str, default='us', choices=['us', 'cn'], help='Region code')
    scan_parser.add_argument('--params-file', type=str, help='JSON file with strategy parameters')
    scan_parser.add_argument('--data-path', type=str, help='Path to Qlib data directory')
    scan_parser.add_argument('--top-k', type=int, default=10, help='Number of top signals to display')
    scan_parser.add_argument('--output-file', type=str, help='Output file for results')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Execute command
        if args.command == 'download':
            command_download_data(args)
        elif args.command == 'run-strategy':
            command_run_strategy(args)
        elif args.command == 'scan-market':
            command_scan_market(args)
        else:
            logger.error(f"Unknown command: {args.command}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
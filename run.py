#!/usr/bin/env python3
"""
Universal CLI for StockScan - A unified entry point for all operations including:
- Backtesting
- Optimization
- Market data downloading
- Strategy running
- Cleaning up
- And more
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import importlib
from typing import Dict, List, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Import the individual script functions
# Note: We'll dynamically import these to avoid circular imports
def get_script_function(script_path: str, function_name: str = 'main') -> Callable:
    """Dynamically import a function from a script."""
    script_name = Path(script_path).stem
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, function_name):
        return getattr(module, function_name)
    else:
        logger.error(f"Function {function_name} not found in {script_path}")
        return None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='StockScan Unified Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Backtest command (from existing run_backtest.py)
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--data-dir', type=str, default='data/market_data',
                                help='Directory containing market data')
    backtest_parser.add_argument('--strategy', type=str, required=True,
                                help='Strategy name to backtest')
    backtest_parser.add_argument('--symbols', type=str, nargs='+',
                                help='Symbols to backtest (e.g., AAPL MSFT)')
    backtest_parser.add_argument('--market', type=str, default='nasdaq',
                                help='Market to backtest (default: nasdaq)')
    backtest_parser.add_argument('--params-file', type=str,
                                help='JSON file with strategy parameters')
    backtest_parser.add_argument('--start-date', type=str, default='2020-01-01',
                                help='Start date for backtesting (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str,
                                help='End date for backtesting (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-capital', type=float, default=10000.0,
                                help='Initial capital for backtesting')
    backtest_parser.add_argument('--output-file', type=str,
                                help='Output file for backtest results')
    
    # Optimization command (from existing run_backtest.py)
    optimize_parser = subparsers.add_parser('optimize', help='Run strategy optimization')
    optimize_parser.add_argument('--data-dir', type=str, default='data/market_data',
                                help='Directory containing market data')
    optimize_parser.add_argument('--strategy', type=str, required=True,
                                help='Strategy name to optimize')
    optimize_parser.add_argument('--symbols', type=str, nargs='+',
                                help='Symbols to optimize for (e.g., AAPL MSFT)')
    optimize_parser.add_argument('--market', type=str, default='nasdaq',
                                help='Market to use for optimization (default: nasdaq)')
    optimize_parser.add_argument('--params-space', type=str, 
                                help='JSON file with parameter space')
    optimize_parser.add_argument('--start-date', type=str, default='2020-01-01',
                                help='Start date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', type=str,
                                help='End date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--initial-capital', type=float, default=10000.0,
                                help='Initial capital for optimization')
    optimize_parser.add_argument('--n-trials', type=int, default=100,
                                help='Number of optimization trials')
    optimize_parser.add_argument('--output-file', type=str,
                                help='Output file for optimization results')
    
    # Download market data command
    download_parser = subparsers.add_parser('download', help='Download market data')
    download_parser.add_argument('--symbols-file', type=str, 
                               help='File containing symbols to download')
    download_parser.add_argument('--symbols', type=str,
                               help='Comma-separated list of symbols to download (e.g., AAPL,MSFT,GOOG)')
    download_parser.add_argument('--save-dir', type=str, default='data/market_data',
                               help='Directory to save market data')
    download_parser.add_argument('--start-date', type=str, default='2018-01-01',
                               help='Start date for data download (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', type=str,
                               help='End date for data download (YYYY-MM-DD)')
    download_parser.add_argument('--interval', type=str, default='1d',
                               help='Data interval (1d, 1h, etc.)')
    
    # Run qlib strategies command
    qlib_parser = subparsers.add_parser('qlib', help='Run qlib strategies')
    qlib_parser.add_argument('--qlib-config', type=str, required=True,
                           help='QLib configuration file')
    qlib_parser.add_argument('--strategy', type=str, required=True,
                           help='QLib strategy to run')
    qlib_parser.add_argument('--symbols', type=str,
                           help='Comma-separated list of symbols to run strategy on')
    qlib_parser.add_argument('--symbols-file', type=str,
                           help='File containing symbols (one per line)')
    qlib_parser.add_argument('--start-date', type=str, default='2020-01-01',
                           help='Start date (YYYY-MM-DD)')
    qlib_parser.add_argument('--end-date', type=str,
                           help='End date (YYYY-MM-DD)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup data and temporary files')
    cleanup_parser.add_argument('--target-dir', type=str, default='.',
                              help='Target directory to clean')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                              help='Perform a dry run without deleting files')
    
    # Scan command (from existing run_backtest.py)
    scan_parser = subparsers.add_parser('scan', help='Scan for stocks matching criteria')
    scan_parser.add_argument('--data-dir', type=str, default='data/market_data',
                           help='Directory containing market data')
    scan_parser.add_argument('--strategy', type=str, required=True,
                           help='Strategy name for scanning')
    scan_parser.add_argument('--market', type=str, default='nasdaq',
                           help='Market to scan (default: nasdaq)')
    scan_parser.add_argument('--params-file', type=str,
                           help='JSON file with strategy parameters')
    scan_parser.add_argument('--output-file', type=str,
                           help='Output file for scan results')
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    if not args.command:
        # No command provided, show help
        print("No command specified. Please use one of the available commands:")
        print("  backtest - Run backtesting")
        print("  optimize - Run strategy optimization")
        print("  scan     - Scan for stocks matching criteria")
        print("  download - Download market data")
        print("  qlib     - Run qlib strategies")
        print("  cleanup  - Cleanup data and temporary files")
        print("\nFor more information, run: python run.py --help")
        sys.exit(1)
    
    try:
        if args.command == 'backtest':
            from run_backtest import run_backtest
            run_backtest(args)
        elif args.command == 'optimize':
            from run_backtest import run_optimization
            # Map market to exchange for backward compatibility
            args.market = args.market if hasattr(args, 'market') else args.exchange
            run_optimization(args)
        elif args.command == 'scan':
            from run_backtest import run_scan
            run_scan(args)
        elif args.command == 'download':
            # Import the download market data functionality
            download_func = get_script_function('scripts/download_market_data.py', 'download_market_data')
            if download_func:
                # Pass relevant arguments
                download_func(
                    symbols_file=args.symbols_file,
                    symbols=args.symbols,
                    save_dir=args.save_dir,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    interval=args.interval
                )
        elif args.command == 'qlib':
            # Import the qlib strategy functionality
            qlib_func = get_script_function('scripts/run_qlib_strategy.py', 'run_qlib_strategy')
            if qlib_func:
                # Convert args to dictionary and pass
                qlib_func(
                    qlib_config=args.qlib_config,
                    strategy=args.strategy,
                    symbols=args.symbols,
                    symbols_file=args.symbols_file,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
        elif args.command == 'cleanup':
            # Import the cleanup functionality
            cleanup_func = get_script_function('scripts/cleanup_root.py')
            if cleanup_func:
                # Pass relevant arguments
                cleanup_func(
                    target_dir=args.target_dir,
                    dry_run=args.dry_run
                )
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
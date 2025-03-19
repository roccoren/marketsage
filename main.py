#!/usr/bin/env python3
"""
StockScan LEAN Backtesting System - Main Entry Point

This script provides a command-line interface to run backtests using the StockScan
backtesting system with the QuantConnect LEAN framework.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import asyncio

from src.tools.configure_storage import main as configure_storage
from src.tools.query_influxdb import main as query_influxdb

# Import LEAN adapter
from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='StockScan LEAN Backtesting System')
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # LEAN backtest command
    lean_parser = subparsers.add_parser('backtest', help='Run backtest using LEAN framework')
    lean_parser.add_argument('--algorithm', required=True, help='Algorithm name to use (e.g., TrendFollowingAlgorithm)')
    lean_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to backtest')
    lean_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    lean_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    lean_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    lean_parser.add_argument('--resolution', choices=['Daily', 'Hour', 'Minute'], default='Daily', help='Data resolution')
    lean_parser.add_argument('--config', help='Path to configuration file')
    lean_parser.add_argument('--data-dir', default='data', help='Data directory')
    lean_parser.add_argument('--results-dir', default='results', help='Results directory')
    
    # Algorithm-specific parameters
    lean_parser.add_argument('--fast-period', type=int, help='Fast period for indicators')
    lean_parser.add_argument('--slow-period', type=int, help='Slow period for indicators')
    lean_parser.add_argument('--rsi-period', type=int, help='RSI period')
    lean_parser.add_argument('--oversold-threshold', type=float, help='Oversold threshold for RSI')
    lean_parser.add_argument('--overbought-threshold', type=float, help='Overbought threshold for RSI')
    lean_parser.add_argument('--stop-loss-pct', type=float, help='Stop loss percentage')
    lean_parser.add_argument('--take-profit-pct', type=float, help='Take profit percentage')
    
    # List LEAN algorithms
    list_algorithms_parser = subparsers.add_parser('list-algorithms', help='List available LEAN algorithms')
    
    # Convert data to LEAN format
    convert_parser = subparsers.add_parser('convert-data', help='Convert market data to LEAN format')
    convert_parser.add_argument('--source-file', required=True, help='Source data file')
    convert_parser.add_argument('--symbol', required=True, help='Symbol name')
    convert_parser.add_argument('--output-file', help='Output file path')
    convert_parser.add_argument('--config', help='Path to configuration file')
    convert_parser.add_argument('--data-dir', help='Data directory')
    
    # Run example
    example_parser = subparsers.add_parser('example', help='Run LEAN example backtest')
    example_parser.add_argument('--algorithm', choices=['trend', 'mean_reversion', 'all'], default='all', 
                               help='Algorithm to run (trend, mean_reversion, or all)')
    
    # Configure storage
    storage_parser = subparsers.add_parser('configure-storage', help='Configure data storage backend')
    storage_parser.add_argument('--use-influxdb', action='store_true', help='Use InfluxDB for data storage')
    storage_parser.add_argument('--use-file-storage', action='store_true', help='Use file-based storage for data')
    storage_parser.add_argument('--influxdb-url', help='InfluxDB server URL')
    storage_parser.add_argument('--influxdb-token', help='InfluxDB authentication token')
    storage_parser.add_argument('--influxdb-org', help='InfluxDB organization')
    storage_parser.add_argument('--influxdb-bucket', help='InfluxDB bucket name')
    
    # Data fetching commands
    fetch_parser = subparsers.add_parser('fetch-data', help='Fetch market data')
    fetch_parser.add_argument('--market', required=True, help='Market to fetch data for (e.g., nyse, nasdaq)')
    fetch_parser.add_argument('--symbols', nargs='+', help='Specific symbols to fetch (optional)')
    fetch_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end-date', help='End date (YYYY-MM-DD), defaults to today')
    fetch_parser.add_argument('--use-enhanced-fetcher', action='store_true', help='Use enhanced market fetcher with additional data')
    
    # Query InfluxDB command
    query_parser = subparsers.add_parser('query-influxdb', help='Query stock data from InfluxDB')
    query_parser.add_argument('--symbol', required=True, help='Stock symbol to query')
    query_parser.add_argument('--market', required=True, help='Market (e.g., nyse, nasdaq)')
    query_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    query_parser.add_argument('--end-date', help='End date (YYYY-MM-DD), defaults to today')
    query_parser.add_argument('--output-format', choices=['csv', 'json', 'plot'], default='plot',
                              help='Output format (csv, json, or plot)')
    query_parser.add_argument('--output-file', help='Output file path')
    query_parser.add_argument('--plot-type', choices=['line', 'candlestick', 'ohlc'], default='line',
                              help='Plot type (line, candlestick, or ohlc)')
    
    # Add verbose flag to all commands
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def run_lean_backtest(args):
    """Run a backtest using the LEAN framework."""
    try:
        # Initialize adapter
        adapter = LeanAdapter(
            config_path=args.config,
            data_folder=args.data_dir,
            results_folder=args.results_dir
        )
        
        # Check if we have algorithms
        if not adapter.get_available_algorithms():
            print("Error: No LEAN algorithms available.")
            print("Make sure LEAN framework is properly installed and algorithms are in the correct location.")
            print("Check src/market_system/lean/algorithms/ directory for algorithm implementations.")
            return
        
        # Get backtest parameters
        parameters = {}
        
        # Add strategy-specific parameters
        if hasattr(args, 'fast_period') and args.fast_period:
            parameters['FastPeriod'] = args.fast_period
        if hasattr(args, 'slow_period') and args.slow_period:
            parameters['SlowPeriod'] = args.slow_period
        if hasattr(args, 'rsi_period') and args.rsi_period:
            parameters['RsiPeriod'] = args.rsi_period
        if hasattr(args, 'oversold_threshold') and args.oversold_threshold:
            parameters['OversoldThreshold'] = args.oversold_threshold
        if hasattr(args, 'overbought_threshold') and args.overbought_threshold:
            parameters['OverboughtThreshold'] = args.overbought_threshold
        if hasattr(args, 'stop_loss_pct') and args.stop_loss_pct:
            parameters['StopLossPercentage'] = args.stop_loss_pct
        if hasattr(args, 'take_profit_pct') and args.take_profit_pct:
            parameters['TakeProfitPercentage'] = args.take_profit_pct
        
        # Run backtest
        results = adapter.run_backtest(
            algorithm_name=args.algorithm,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date or datetime.now().strftime('%Y-%m-%d'),
            initial_capital=args.capital or 100000,
            resolution=args.resolution or 'Daily',
            parameters=parameters
        )
        
        # Display results
        print("\nBacktest Results:")
        print(f"Algorithm: {results['algorithm_name']}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['metrics']['total_return']:.2%}")
        print(f"Annual Return: {results['metrics']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['metrics']['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Runtime: {results['runtime']}")
        
        # Print result files
        if results['result_files']:
            print("\nResult Files:")
            for name, path in results['result_files'].items():
                print(f"- {name}: {path}")
    except Exception as e:
        print(f"Error running backtest: {e}")
        print("Make sure LEAN framework is properly installed and required data is available.")
        
    return


def list_lean_algorithms(args):
    """List available LEAN algorithms."""
    try:
        adapter = LeanAdapter()
        algorithms = adapter.get_available_algorithms()
        
        if not algorithms:
            print("No LEAN algorithms found.")
            return
        
        print("Available LEAN algorithms:")
        for algo in algorithms:
            print(f"  - {algo}")
    except Exception as e:
        print(f"Error: {e}")
        print("Could not list algorithms. Make sure LEAN framework is installed.")
        print("To install LEAN, follow the instructions at: https://github.com/QuantConnect/Lean")
        
    return


def convert_data_to_lean(args):
    """Convert market data to LEAN format."""
    try:
        # Initialize adapter
        adapter = LeanAdapter(
            config_path=args.config,
            data_folder=args.data_dir,
            results_folder=None
        )
        
        # Validate input
        if not os.path.exists(args.source_file):
            print(f"Error: Source file '{args.source_file}' does not exist.")
            return
        
        # Convert data
        output_path = adapter.convert_data_to_lean(
            data_path=args.source_file,
            symbol=args.symbol,
            output_path=args.output_file
        )
        
        print(f"Data successfully converted to LEAN format.")
        print(f"Output file: {output_path}")
    except Exception as e:
        print(f"Error converting data: {e}")
        print("Make sure the source file has the required columns: timestamp, open, high, low, close, volume")
    
    return


def run_example(args):
    """Run LEAN example backtest."""
    import examples.lean_backtest_example as example
    
    if args.algorithm == 'trend':
        example.run_trend_following_backtest()
    elif args.algorithm == 'mean_reversion':
        example.run_mean_reversion_backtest()
    else:  # 'all'
        example.run_all_backtests()


def fetch_market_data(args):
    """Fetch market data based on command-line arguments."""
    from src.market_data.core.market_fetcher import ConsolidatedMarketFetcher
    from src.market_data.providers.enhanced_fetcher import EnhancedMarketFetcher
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    async def fetch_data():
        if args.use_enhanced_fetcher:
            # Use enhanced fetcher with additional data fields
            fetcher = EnhancedMarketFetcher()
        else:
            # Use standard fetcher
            fetcher = ConsolidatedMarketFetcher()
        
        # Fetch the data
        await fetcher.fetch_market_data(
            market=args.market,
            start_date=start_date,
            end_date=end_date,
            symbols=args.symbols
        )
        
        print(f"Data fetching completed for {args.market}")
    
    # Run the async function
    asyncio.run(fetch_data())


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.command == 'backtest':
        run_lean_backtest(args)
    elif args.command == 'list-algorithms':
        list_lean_algorithms(args)
    elif args.command == 'convert-data':
        convert_data_to_lean(args)
    elif args.command == 'example':
        run_example(args)
    elif args.command == 'fetch-data':
        fetch_market_data(args)
    elif args.command == 'configure-storage':
        # Pass arguments to the configure_storage function
        from argparse import Namespace
        storage_args = Namespace(
            use_influxdb=args.use_influxdb,
            use_file_storage=args.use_file_storage,
            influxdb_url=args.influxdb_url,
            influxdb_token=args.influxdb_token,
            influxdb_org=args.influxdb_org,
            influxdb_bucket=args.influxdb_bucket
        )
        configure_storage(storage_args)
    elif args.command == 'query-influxdb':
        # Call the query_influxdb function
        query_influxdb()
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main() 
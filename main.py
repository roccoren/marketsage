#!/usr/bin/env python3
"""
StockScan Backtesting System - Main Entry Point

This script provides a command-line interface to run backtests using the StockScan
backtesting system.
"""

import argparse
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.market_system.backtesting.example import run_backtest
from src.market_system.backtesting.backtest_engine import BacktestEngine
from src.market_system.backtesting.data_provider import DataProvider
from src.market_system.backtesting.portfolio_manager import PortfolioManager
from src.market_system.backtesting.execution_handler import ExecutionHandler
from src.market_system.backtesting.strategies import TrendFollowingStrategy
from src.tools.configure_storage import main as configure_storage
from src.tools.query_influxdb import main as query_influxdb


def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='StockScan Backtesting System')
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run example backtest
    example_parser = subparsers.add_parser('example', help='Run example backtest')
    
    # Run custom backtest
    backtest_parser = subparsers.add_parser('backtest', help='Run custom backtest')
    backtest_parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to backtest')
    backtest_parser.add_argument('--timeframe', default='1d', help='Timeframe for backtest (e.g., 1d, 1h)')
    backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    backtest_parser.add_argument('--risk', type=float, default=1.0, help='Risk percentage per trade')
    backtest_parser.add_argument('--fast-ma', type=int, default=20, help='Fast moving average period')
    backtest_parser.add_argument('--slow-ma', type=int, default=50, help='Slow moving average period')
    
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


def run_custom_backtest(args):
    """Run a custom backtest based on command line arguments."""
    # Set up paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dates
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else (end_date - timedelta(days=365))
    
    # Set up report name
    report_name = args.report_name or f"backtest_{args.symbols[0]}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    # Initialize components
    data_provider = DataProvider(str(data_dir))
    strategy = TrendFollowingStrategy(
        fast_period=args.fast_ma,
        slow_period=args.slow_ma
    )
    portfolio_manager = PortfolioManager(initial_capital=Decimal(str(args.capital)))
    execution_handler = ExecutionHandler(
        slippage_model="percentage",
        slippage_params={"percentage": 0.001},
        commission_model="fixed",
        commission_params={"fixed_amount": Decimal("5.0")}
    )
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=Decimal(str(args.capital)),
        start_date=start_date,
        end_date=end_date,
        data_provider=data_provider,
        strategies=[strategy],
        portfolio_manager=portfolio_manager,
        execution_handler=execution_handler
    )
    
    # Run backtest
    results = engine.run(
        symbols=args.symbols,
        timeframe=args.timeframe,
        risk_percentage=args.risk,
        show_progress=True
    )
    
    # Generate report
    report_paths = engine.generate_report(
        output_dir=str(output_dir),
        report_name=report_name
    )
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Strategy: {results['strategy_name']}")
    print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
    print(f"Initial Capital: ${float(results['initial_capital']):,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    
    # Print metrics (if available)
    metrics = results.get('metrics', {})
    print(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
    
    if metrics.get('total_trades', 0) > 0:
        print(f"Win Rate: {metrics.get('win_rate', 0.0):.2%}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
    else:
        print("No trades executed during the backtest period.")
    
    print("\nReport generated at:")
    for format_name, path in report_paths.items():
        print(f"- {format_name}: {path}")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.command == 'example':
        run_backtest()
    elif args.command == 'backtest':
        run_custom_backtest(args)
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


def fetch_market_data(args):
    """Fetch market data based on command-line arguments."""
    import asyncio
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


if __name__ == "__main__":
    main() 
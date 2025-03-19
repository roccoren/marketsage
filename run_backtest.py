#!/usr/bin/env python3
"""
Universal CLI for StockScan - Run backtest, optimization, scan stocks, and more
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import pandas as pd
import importlib
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from stockscan.utils.exchange_data_loader import ExchangeDataLoader
from stockscan.strategies.backtester import Backtester
from stockscan.utils.backtest_formatter import display_results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='StockScan Universal CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run backtest:      python run_backtest.py backtest --strategy macd --exchange NYSE --symbol AAPL
  Run optimization:  python run_backtest.py optimize --strategy multi_indicator --exchange NYSE --symbols AAPL,MSFT
  Run scan:          python run_backtest.py scan --exchange NYSE --limit 10
  """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--strategy', type=str, required=True, 
                               help='Strategy name (e.g., macd, multi_indicator)')
    backtest_parser.add_argument('--exchange', type=str, required=True, 
                               choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX'],
                               help='Exchange name')
    backtest_parser.add_argument('--symbol', type=str, required=True,
                               help='Stock symbol to backtest')
    backtest_parser.add_argument('--start-date', type=str, default='2020-01-01',
                              help='Start date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, default=None,
                              help='End date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=100000.0,
                              help='Initial capital')
    backtest_parser.add_argument('--position-size', type=float, default=1.0,
                              help='Position size as decimal (0.0-1.0), or fixed size if > 1.0')
    backtest_parser.add_argument('--commission', type=float, default=0.0,
                              help='Commission per trade (% as decimal)')
    backtest_parser.add_argument('--slippage', type=float, default=0.0,
                              help='Slippage per trade (% as decimal)')
    backtest_parser.add_argument('--params-file', type=str, default=None,
                              help='Path to JSON file with strategy parameters')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run strategy optimization')
    optimize_parser.add_argument('--strategy', type=str, required=True,
                               help='Strategy name to optimize')
    optimize_parser.add_argument('--exchange', type=str, required=True,
                              choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX'],
                              help='Exchange name')
    optimize_parser.add_argument('--symbols', type=str, required=True,
                              help='Comma-separated list of symbols to optimize on')
    optimize_parser.add_argument('--start-date', type=str, default='2020-01-01',
                              help='Start date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', type=str, default=None,
                              help='End date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--metric', type=str, default='sharpe_ratio',
                              choices=['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'net_profit'],
                              help='Metric to optimize for')
    optimize_parser.add_argument('--iterations', type=int, default=10,
                              help='Number of optimization iterations')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for trading signals')
    scan_parser.add_argument('--exchange', type=str, required=True,
                           choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX', 'ALL'],
                           help='Exchange name or ALL for all exchanges')
    scan_parser.add_argument('--strategy', type=str, default='multi_indicator',
                           help='Strategy name (default: multi_indicator)')
    scan_parser.add_argument('--limit', type=int, default=None,
                           help='Limit number of symbols to scan')
    scan_parser.add_argument('--days', type=int, default=60,
                           help='Number of days of data to analyze')
    
    return parser.parse_args()


def load_strategy(strategy_name: str, params_file: Optional[str] = None) -> Any:
    """Load strategy class by name"""
    params = {}
    if params_file and os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
                logger.info(f"Loaded parameters from {params_file}")
        except Exception as e:
            logger.error(f"Error loading parameters file: {str(e)}")
    
    try:
        strategy_map = {
            'macd': 'stockscan.strategies.technical.macd',
            'multi_indicator': 'stockscan.strategies.technical.multi_indicator_strategy',
            'crossover': 'stockscan.strategies.crossover',
            'combined_signal': 'stockscan.strategies.combined_signal',
            'advanced_multi': 'stockscan.strategies.advanced_multi_strategy',
            'optimized_multi': 'stockscan.strategies.optimized_multi_strategy',
            'adx_macd_cci': 'stockscan.strategies.technical.strategy_adx_macd_cci',
            'atr_moving_average': 'stockscan.strategies.strategy_atr_moving_average',
            'obv': 'stockscan.strategies.strategy_obv',
            'rsi_stochastic': 'stockscan.strategies.technical.strategy_rsi_stochastic'
        }
        
        if strategy_name not in strategy_map:
            logger.error(f"Unknown strategy: {strategy_name}")
            logger.info(f"Available strategies: {', '.join(strategy_map.keys())}")
            return None
            
        module_name = strategy_map[strategy_name]
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
        
        # Import the module dynamically
        module = __import__(module_name, fromlist=[class_name])
        strategy_class = getattr(module, class_name)
        
        # Instantiate with parameters if provided
        return strategy_class(params=params)
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading strategy {strategy_name}: {str(e)}")
        return None


def run_backtest(args) -> None:
    """Run backtest with given arguments"""
    symbol = args.symbols[0] if isinstance(args.symbols, list) else args.symbols
    market = args.market
    
    logger.info(f"Running backtest for {symbol} on {market} using {args.strategy} strategy")
    
    # Load strategy
    strategy = load_strategy(args.strategy, args.params_file)
    if not strategy:
        return
    
    # Load data
    data_loader = ExchangeDataLoader()
    data = data_loader.load_stock_data(market, symbol)
    
    if data is None or len(data) < 50:
        logger.error(f"Insufficient data for {symbol} ({market})")
        return
    
    # Set up backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=args.capital if hasattr(args, 'capital') else 10000,
        position_size=args.position_size if hasattr(args, 'position_size') else 1.0,
        commission=args.commission if hasattr(args, 'commission') else 0.001,
        slippage=args.slippage if hasattr(args, 'slippage') else 0.001
    )
    
    # Run backtest
    result = backtester.run(
        data=data,
        symbol=symbol,
        start_date=args.start_date if args.start_date else None,
        end_date=args.end_date if args.end_date else None
    )
    
    if result:
        # Save results
        output_dir = Path("output/backtest")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{symbol}_{args.strategy}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            metrics = result.metrics.copy()
            for k, v in metrics.items():
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    metrics[k] = v.to_dict()
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Display results
        display_results(symbol, market, result)
    else:
        logger.error("Backtest failed to produce results")


def run_optimization(args) -> None:
    """Run strategy optimization with given arguments"""
    from stockscan.strategies.optimizer import StrategyOptimizer
    
    # Normalize symbols to a list
    if isinstance(args.symbols, list):
        symbols = args.symbols
    elif isinstance(args.symbols, str):
        symbols = args.symbols.split(',')
    else:
        logger.error(f"Invalid symbols format: {args.symbols}")
        return
    
    logger.info(f"Running optimization for {args.strategy} on {symbols[0]}")

    # Load strategy class
    try:
        strategy_module = importlib.import_module(f"stockscan.strategies.{args.strategy}")
        strategy_name = args.strategy.replace('_', ' ')
        strategy_class_name = ''.join(word.capitalize() for word in strategy_name.split()) + 'Strategy'
        
        strategy_class = getattr(strategy_module, strategy_class_name)
        
        # Load data for first symbol for parameter grid
        data_loader = ExchangeDataLoader()
        data = data_loader.load_stock_data(args.market, symbols[0])
        
        if data is None or len(data) < 100:
            logger.error(f"Insufficient data for optimization: {symbols[0]}")
            return
            
        # Get parameter grid from strategy
        param_grid = strategy_class.get_param_grid() if hasattr(strategy_class, 'get_param_grid') else {}
        
        if not param_grid:
            # Use default parameter grid for supported strategies
            if args.strategy == 'macd':
                param_grid = {
                    'fast_period': [8, 12, 14],
                    'slow_period': [21, 26, 30],
                    'signal_period': [7, 9, 11],
                    'rsi_period': [7, 14, 21],
                    'rsi_overbought': [65, 70, 75],
                    'rsi_oversold': [25, 30, 35],
                    'volume_ma_period': [5, 10, 20]
                }
            else:
                logger.warning(f"No parameter grid defined for {args.strategy}, using empty grid")
        
        # Create optimizer
        n_trials = args.n_trials if hasattr(args, 'n_trials') else (args.iterations if hasattr(args, 'iterations') else 10)
        optimizer = StrategyOptimizer(
            strategy_class=strategy_class,
            param_grid=param_grid,
            data=data,
            train_size=0.7,
            metric=args.metric if hasattr(args, 'metric') else 'sharpe_ratio'
        )
        
        # Run optimization
        results = optimizer.optimize(iterations=n_trials, initial_capital=100000)
        
        if results['best_params']:
            # Save results
            output_dir = Path("output/optimization")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_file = output_dir / f"optimal_params_{args.strategy}_{timestamp}.json"
            report_file = output_dir / f"optimization_report_{args.strategy}_{timestamp}.csv"
            
            with open(params_file, 'w') as f:
                json.dump(results['best_params'], f, indent=2)
                
            report = optimizer.get_optimization_report()
            report.to_csv(report_file)
            
            # Display optimization results
            logger.info("Optimization results:")
            logger.info(f"Best parameters: {results['best_params']}")
            metric_name = args.metric if hasattr(args, 'metric') else 'sharpe_ratio'
            logger.info(f"Best {metric_name}: {results['best_train_score']:.4f}")
            
            # Only display test score if it exists
            if results.get('best_test_score') is not None:
                logger.info(f"Test {metric_name}: {results['best_test_score']:.4f}")
            else:
                logger.info(f"Test {metric_name}: Not available")
            
            logger.info(f"Parameters saved to {params_file}")
            logger.info(f"Optimization report saved to {report_file}")
            
            # Run backtest with optimized parameters
            logger.info("Running backtest with optimized parameters...")
            for symbol in symbols:
                data = data_loader.load_stock_data(args.market, symbol)
                if data is None or len(data) < 50:
                    logger.warning(f"Skipping {symbol}: insufficient data")
                    continue
                    
                strategy = strategy_class(params=results['best_params'])
                backtester = Backtester(strategy)
                backtest_result = backtester.run(
                    data=data,
                    symbol=symbol,
                    start_date=args.start_date if args.start_date else None,
                    end_date=args.end_date if args.end_date else None
                )
                
                if backtest_result:
                    display_results(symbol, args.market, backtest_result)
        else:
            logger.error("Optimization failed to find valid parameters")
            
    except (ImportError, AttributeError) as e:
        logger.error(f"Error during optimization: {str(e)}")


def run_scan(args) -> None:
    """Run stock scanner with given arguments"""
    logger.info(f"Running scan with {args.strategy} strategy")
    
    # Load strategy
    strategy = load_strategy(args.strategy, args.params_file)
    if not strategy:
        return
        
    from stockscan.scanner.stock_scanner import StockScanner
    
    # Prepare scanner config
    exchanges = ['NYSE', 'NASDAQ', 'ASX', 'HKEX'] if args.market == 'ALL' else [args.market]
    
    config = {
        'exchanges': exchanges,
        'limit': args.limit
    }
    
    # Create and run scanner
    scanner = StockScanner(config)
    scanner.register_strategy(strategy)
    
    try:
        # Run scan synchronously
        import asyncio
        asyncio.run(scanner.scan(days=args.days))
        logger.info("Scan completed successfully")
    except Exception as e:
        logger.error(f"Error during scan: {str(e)}")


def main() -> None:
    """Main function"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    
    try:
        if args.command == 'backtest':
            run_backtest(args)
        elif args.command == 'optimize':
            run_optimization(args)
        elif args.command == 'scan':
            run_scan(args)
        else:
            logger.error(f"Unknown command: {args.command}")
    except Exception as e:
        logger.exception(f"Error during execution: {str(e)}")


if __name__ == "__main__":
    main()
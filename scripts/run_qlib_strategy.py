#!/usr/bin/env python3
"""
Script to run strategy on stock data using Qlib
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path to import local modules
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from stockscan.core.qlib_strategy import QlibStrategyRunner

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

def load_strategy_params(params_file: str) -> Dict[str, Any]:
    """Load strategy parameters from a JSON file."""
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded parameters from {params_file}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_file}: {str(e)}")
        return {}

def run_strategy_for_symbol(
    strategy_runner: QlibStrategyRunner,
    strategy: str,
    symbol: str,
    data_dir: str,
    start_date: str,
    end_date: str,
    params: Dict[str, Any],
    output_dir: str = None
) -> Dict[str, Any]:
    """Run strategy for a single symbol."""
    logger.info(f"Running {strategy} strategy for {symbol}")
    
    result = None
    if strategy.lower() == 'macd':
        result = strategy_runner.run_macd_strategy(
            symbol=symbol,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            fast_period=params.get('fast_period', 12),
            slow_period=params.get('slow_period', 26),
            signal_period=params.get('signal_period', 9)
        )
    elif strategy.lower() == 'rsi':
        result = strategy_runner.run_rsi_strategy(
            symbol=symbol,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            rsi_period=params.get('rsi_period', 14),
            overbought=params.get('overbought', 70),
            oversold=params.get('oversold', 30)
        )
    else:
        logger.error(f"Unknown strategy: {strategy}")
        return None
    
    # Save result if output directory is specified
    if output_dir and result and result.get('success'):
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{symbol}_{strategy}_{timestamp}.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    return result

def print_strategy_result(result: Dict[str, Any]):
    """Print strategy result in a user-friendly format."""
    if not result or not result.get('success'):
        logger.error(f"Strategy failed: {result.get('message', 'Unknown error')}")
        return
    
    print("\n" + "=" * 60)
    print(f"Strategy Results for {result['symbol']}")
    print("=" * 60)
    print(f"Period: {result['period']}")
    print(f"Strategy: {result.get('strategy', '')}")
    print(f"Current Signal: {result['current_signal'].upper()}")
    
    # Print metrics
    metrics = result.get('metrics', {})
    if metrics:
        print("\nPerformance Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Print recent signals
    signals = result.get('signals', [])
    if signals:
        recent_signals = signals[-5:] if len(signals) > 5 else signals
        print("\nRecent Signals:")
        print("-" * 40)
        for signal in recent_signals:
            print(f"{signal['date']} - {signal['type'].upper()} at ${signal['price']:.2f}")
    
    print("\n")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run strategy on stock data using Qlib'
    )
    parser.add_argument('--strategy', type=str, required=True, choices=['macd', 'rsi'],
                      help='Strategy to run')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--symbols-file', type=str, help='File containing symbols (one per line)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--params-file', type=str, help='JSON file with strategy parameters')
    parser.add_argument('--region', type=str, default='us', choices=['us', 'cn'], help='Region code')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
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
        
        # Load strategy parameters
        params = {}
        if args.params_file:
            params = load_strategy_params(args.params_file)
        
        # Set default end date if not provided
        end_date = args.end_date if args.end_date else datetime.now().strftime("%Y-%m-%d")
        
        # Initialize strategy runner
        strategy_runner = QlibStrategyRunner(region=args.region)
        
        # Initialize Qlib
        strategy_runner.init_qlib()
        
        # Run strategy for each symbol
        results = []
        for symbol in symbols:
            result = run_strategy_for_symbol(
                strategy_runner=strategy_runner,
                strategy=args.strategy,
                symbol=symbol,
                data_dir=args.data_dir,
                start_date=args.start_date,
                end_date=end_date,
                params=params,
                output_dir=args.output_dir
            )
            
            if result and result.get('success'):
                results.append(result)
                print_strategy_result(result)
        
        # Summary
        logger.info(f"Completed running {args.strategy} strategy for {len(results)} symbols")
        
        # Print buy signals
        buy_signals = [r for r in results if r.get('current_signal') == 'buy']
        if buy_signals:
            print("\nBuy Signals:")
            for signal in buy_signals:
                print(f"{signal['symbol']}: Current Price ${signal['metrics'].get('current_price', 0):.2f}")
        
        # Print sell signals
        sell_signals = [r for r in results if r.get('current_signal') == 'sell']
        if sell_signals:
            print("\nSell Signals:")
            for signal in sell_signals:
                print(f"{signal['symbol']}: Current Price ${signal['metrics'].get('current_price', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def run_qlib_strategy(qlib_config=None, strategy=None, start_date=None, end_date=None, symbols=None, symbols_file=None, data_dir=None, params_file=None, region=None, output_dir=None):
    """Run QLib strategy with provided parameters - callable from external scripts."""
    try:
        # Set defaults
        strategy = strategy or 'macd'
        start_date = start_date or '2023-01-01'
        region = region or 'us'
        data_dir = data_dir or 'data'
        
        # Load symbols
        symbol_list = []
        if symbols_file:
            symbol_list = load_symbols_from_file(symbols_file)
        elif symbols:
            if isinstance(symbols, str):
                symbol_list = symbols.split(',')
            elif isinstance(symbols, list):
                symbol_list = symbols
        
        if not symbol_list:
            logger.error("No symbols specified. Use symbols or symbols_file")
            return
        
        # Load strategy parameters
        params = {}
        if params_file:
            params = load_strategy_params(params_file)
        
        # Set default end date if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize strategy runner
        strategy_runner = QlibStrategyRunner(region=region, config_file=qlib_config)
        
        # Initialize Qlib
        strategy_runner.init_qlib()
        
        # Run strategy for each symbol
        results = []
        for symbol in symbol_list:
            result = run_strategy_for_symbol(
                strategy_runner=strategy_runner,
                strategy=strategy,
                symbol=symbol,
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date,
                params=params,
                output_dir=output_dir
            )
            
            if result and result.get('success'):
                results.append(result)
                print_strategy_result(result)
        
        # Summary
        logger.info(f"Completed running {strategy} strategy for {len(results)} symbols")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 
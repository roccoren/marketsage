#!/usr/bin/env python
"""
Symbol-Specific Parameters Test Script

This script tests the MovingAverageCrossoverStrategy with symbol-specific parameters.
It first creates dummy parameters for a few test symbols, then runs a backtest using these parameters.
"""

import json
import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_system.market_backtest import run_market_backtest
from src.market_system.storage import ParameterStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config/market/mid-mrk-momentum-master.json"
START_DATE = "2024-01-01"
END_DATE = "2024-03-10"
TIMEFRAME = "1d"
EXECUTION_MODE = "parallel"
WORKERS = 4
PARAM_DB_PATH = "data/strategy_params.db"
REPORT_DIR = "reports/backtest/symbol_params_test"

def load_symbols(config_file, max_symbols=5):
    """Load symbols from configuration file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            symbols = config.get('symbols', [])[:max_symbols]
            
        if not symbols:
            logger.error(f"No symbols found in configuration file: {config_file}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(symbols)} symbols for testing")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {config_file}: {str(e)}")
        sys.exit(1)

def create_test_parameters(symbols):
    """Create test parameters for each symbol"""
    param_storage = ParameterStorage(PARAM_DB_PATH)
    strategy_name = "MAC_10_30"
    
    # Define different parameters for each symbol
    symbol_params = {
        symbols[0]: {
            'fast_period': 5,           # Very fast
            'slow_period': 20,          # Standard slow
            'volume_threshold': 1.2,    # Lower volume requirement
            'rsi_period': 9,            # Shorter RSI
            'rsi_overbought': 75,       # Higher overbought
            'rsi_oversold': 25,         # Lower oversold
            'signal_threshold': 0.001   # Small threshold
        },
        symbols[1]: {
            'fast_period': 15,          # Slower fast
            'slow_period': 40,          # Very slow
            'volume_threshold': 2.0,    # Higher volume requirement
            'rsi_period': 21,           # Longer RSI
            'rsi_overbought': 65,       # Lower overbought
            'rsi_oversold': 35,         # Higher oversold
            'signal_threshold': 0.002   # Larger threshold
        }
    }
    
    # For remaining symbols, use default parameters
    for symbol in symbols[2:]:
        symbol_params[symbol] = {
            'fast_period': 10,          # Default fast
            'slow_period': 30,          # Default slow
            'volume_threshold': 1.5,    # Default volume
            'rsi_period': 14,           # Default RSI
            'rsi_overbought': 70,       # Default overbought
            'rsi_oversold': 30,         # Default oversold
            'signal_threshold': 0.0     # Default threshold
        }
    
    # Store parameters in the database
    for symbol, params in symbol_params.items():
        param_storage.set_parameters(strategy_name, symbol, params)
        logger.info(f"Created test parameters for {symbol}: {params}")
    
    return symbols

def run_test_backtest(symbols):
    """Run a backtest with symbol-specific parameters"""
    strategies = ["moving_average_crossover"]
    
    # Create report directory
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Run backtest with symbol-specific parameters
    logger.info("Running backtest with symbol-specific parameters")
    
    # Process each symbol sequentially to avoid multiprocessing issues
    results = {}
    
    # Initialize DataProvider once for all symbols
    from src.market_system.backtesting.data_provider import DataProvider
    from src.market_system.backtesting.portfolio_manager import PortfolioManager
    from src.market_system.backtesting.execution_handler import ExecutionHandler
    from src.market_system.backtesting.backtest_engine import BacktestEngine
    from src.market_system.backtesting.strategies import MovingAverageCrossoverStrategy
    from src.market_system.backtesting.report_generator import ReportGenerator
    from src.market_system.domain.models import MarketData
    from datetime import datetime
    from decimal import Decimal
    
    # Convert dates
    start_date_obj = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_date_obj = datetime.strptime(END_DATE, '%Y-%m-%d')
    
    # Set up data provider
    data_provider = DataProvider(use_influxdb=True)
    
    # Get all market data at once
    market_data_dict = data_provider.get_data(symbols, start_date_obj, end_date_obj, TIMEFRAME)
    
    for symbol in symbols:
        try:
            logger.info(f"Processing backtest for {symbol}")
            
            # Skip if no data for this symbol
            if symbol not in market_data_dict or market_data_dict[symbol].empty:
                logger.warning(f"No data available for {symbol}, skipping")
                results[symbol] = {}
                continue
            
            # Set up portfolio manager
            portfolio_manager = PortfolioManager(Decimal('100000'))
            
            # Set up execution handler
            execution_handler = ExecutionHandler(
                slippage_model='percentage',
                slippage_params={'percentage': 0.001},
                commission_model='percentage',
                commission_params={'percentage': 0.001, 'min_commission': Decimal('1.0')}
            )
            
            # Set up the strategy with symbol-specific parameters
            strategy = MovingAverageCrossoverStrategy(
                priority=1,
                use_symbol_specific_params=True,
                param_db_path=PARAM_DB_PATH
            )
            
            # Set up backtest engine with correct parameters
            backtest_engine = BacktestEngine(
                initial_capital=Decimal('100000'),
                start_date=start_date_obj,
                end_date=end_date_obj,
                data_provider=data_provider,
                strategies=[strategy],
                portfolio_manager=portfolio_manager,
                execution_handler=execution_handler
            )
            
            # Run backtest using the run method
            logger.info(f"Starting backtest for {symbol}")
            start_time = datetime.now()
            
            # Run backtest with just this one symbol
            backtest_results = backtest_engine.run(
                symbols=[symbol],
                timeframe=TIMEFRAME,
                risk_percentage=Decimal('0.01'),  # 1% risk per trade
                parallel=False,
                show_progress=False
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Backtest for {symbol} completed in {execution_time:.2f} seconds")
            
            # Generate reports
            report_dir = f"reports/backtest/symbol_params_test/{symbol}"
            os.makedirs(report_dir, exist_ok=True)
            report_generator = ReportGenerator(report_dir)
            try:
                report_generator.generate_report(backtest_results, f"{symbol}_report")
            except Exception as e:
                logger.error(f"Error in backtest for {symbol}: {e}")
                logger.error(traceback.format_exc())
            
            # Extract metrics
            metrics = backtest_results.get('metrics', {}) if backtest_results else {}
            metrics['execution_time'] = execution_time
            
            results[symbol] = metrics
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results[symbol] = {}
    
    # Generate a simple summary
    logger.info("Backtest results:")
    
    for symbol, metrics in results.items():
        logger.info(f"{symbol}:")
        logger.info(f"  Return: {metrics.get('total_return', 0) * 100:.2f}%")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    return results

def main():
    """Main function to test symbol-specific parameters"""
    # Step 1: Load symbols
    symbols = load_symbols(CONFIG_FILE)
    
    # Step 2: Create test parameters
    symbols = create_test_parameters(symbols)
    
    # Step 3: Run backtest with symbol-specific parameters
    run_test_backtest(symbols)
    
    logger.info("Symbol-specific parameters test completed")
    logger.info(f"Check reports at: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    logger.info(f"Total runtime: {end_time - start_time}") 
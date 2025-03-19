#!/usr/bin/env python3
"""
Market-wide Backtesting Script

This script runs a backtest for all symbols in a specified market using various trading strategies
for the specified time period. Supported markets include HKEX, NASDAQ, NYSE, and ASX.
"""

import os
import sys
import logging
import asyncio
import time
import psutil
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import pandas as pd
import concurrent.futures
import argparse
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Dict, Optional, Union, Tuple
from influxdb_client import InfluxDBClient
import traceback
import json

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

from .backtesting.backtest_engine import BacktestEngine
from .backtesting.data_provider import DataProvider, load_storage_config
from .backtesting.portfolio_manager import PortfolioManager
from .backtesting.execution_handler import ExecutionHandler
from .backtesting.strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    VolatilityBreakoutStrategy,
    VolumeSpikeStrategy
)
from src.market_system.backtesting.report_generator import ReportGenerator
from .strategies.moving_average_crossover import MovingAverageCrossover
from .strategies.rsi_strategy import RSIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_backtest.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run market backtests')
    
    # Input source options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--symbol', type=str, help='Symbol(s) to backtest (comma-separated)')
    input_group.add_argument('--symbol-file', type=str, help='File containing symbols to backtest')
    
    # Time range options
    time_group = parser.add_argument_group('Time Range')
    time_group.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    time_group.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    time_group.add_argument('--timeframe', type=str, default='1d', help='Timeframe (1d, 1h, etc.)')
    
    # Strategy options
    strategy_group = parser.add_argument_group('Strategy Options')
    strategy_group.add_argument('--strategies', type=str, default='moving_average_crossover', 
                               help='Strategy to use (comma-separated for multiple)')
    
    # Strategy parameters
    params_group = parser.add_argument_group('Strategy Parameters')
    params_group.add_argument('--fast-period', type=int, default=12, help='Fast period for MA crossover')
    params_group.add_argument('--slow-period', type=int, default=26, help='Slow period for MA crossover')
    params_group.add_argument('--rsi-period', type=int, default=14, help='Period for RSI calculation')
    params_group.add_argument('--oversold-threshold', type=int, default=30, help='Oversold threshold for RSI')
    params_group.add_argument('--overbought-threshold', type=int, default=70, help='Overbought threshold for RSI')
    params_group.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss percentage')
    params_group.add_argument('--take-profit', type=float, default=0.05, help='Take profit percentage')
    params_group.add_argument('--position-size', type=float, default=0.1, help='Position size as fraction of portfolio')
    params_group.add_argument('--macd-fast', type=int, default=12, help='Fast period for MACD')
    params_group.add_argument('--macd-slow', type=int, default=26, help='Slow period for MACD')
    params_group.add_argument('--macd-signal', type=int, default=9, help='Signal period for MACD')
    params_group.add_argument('--atr-period', type=int, default=14, help='Period for ATR calculation')
    
    # Advanced strategy options
    advanced_group = parser.add_argument_group('Advanced Strategy Options')
    advanced_group.add_argument('--adaptive-params', action='store_true', help='Use adaptive parameters')
    advanced_group.add_argument('--use-market-regime', action='store_true', help='Use market regime detection')
    advanced_group.add_argument('--use-symbol-specific-params', action='store_true', help='Use symbol-specific parameters')
    advanced_group.add_argument('--param-db-path', type=str, default='data/strategy_params.db', help='Path to parameter database')
    
    # Execution options
    execution_group = parser.add_argument_group('Execution Options')
    execution_group.add_argument('--execution-mode', type=str, default='parallel', choices=['sequential', 'parallel', 'async'], 
                                help='Execution mode')
    execution_group.add_argument('--workers', type=int, default=None, help='Number of workers for parallel execution')
    execution_group.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    execution_group.add_argument('--slippage-value', type=float, default=0.001, help='Slippage value')
    execution_group.add_argument('--commission-value', type=float, default=0.001, help='Commission value')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--generate-reports', action='store_true', help='Generate reports')
    output_group.add_argument('--show-progress', action='store_true', help='Show progress during execution')
    output_group.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Logging level')
    output_group.add_argument('--verbose', action='store_true', help='Print verbose information')
    
    args = parser.parse_args()
    
    # Process strategies list
    if args.strategies and ',' in args.strategies:
        args.strategies = [s.strip() for s in args.strategies.split(',')]
    else:
        args.strategies = [args.strategies]
    
    return args

def get_market_symbols(data_dir: Path, market: str = "hkex") -> list:
    """Get all available symbols from the specified market data directory."""
    market_dir = data_dir / market
    if not market_dir.exists():
        logger.error(f"{market.upper()} data directory not found: {market_dir}")
        return []
    
    symbols = [d.name for d in market_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(symbols)} {market.upper()} symbols")
    return symbols

def initialize_strategies(strategy_names: List[str], **kwargs) -> List:
    """Initialize trading strategies."""
    strategies = []
    
    # Get parameters from kwargs or use defaults
    fast_period = kwargs.get('fast_period', 10)
    slow_period = kwargs.get('slow_period', 30)
    rsi_period = kwargs.get('rsi_period', 14)
    oversold_threshold = kwargs.get('oversold_threshold', 30)  # More conservative (was 25)
    overbought_threshold = kwargs.get('overbought_threshold', 70)  # More conservative (was 75)
    volume_threshold = kwargs.get('volume_threshold', 1.5)
    volatility_threshold = kwargs.get('volatility_threshold', 1.5)  # More conservative (was 2.0)
    consolidation_threshold = kwargs.get('consolidation_threshold', 0.3)  # More conservative (was 0.4)
    
    # Additional MAC parameters with defaults
    macd_fast = kwargs.get('macd_fast', 12)
    macd_slow = kwargs.get('macd_slow', 26)
    macd_signal = kwargs.get('macd_signal', 9)
    atr_period = kwargs.get('atr_period', 14)
    adaptive_params = kwargs.get('adaptive_params', True)
    use_market_regime = kwargs.get('use_market_regime', True)
    use_symbol_specific_params = kwargs.get('use_symbol_specific_params', False)
    param_db_path = kwargs.get('param_db_path', "data/strategy_params.db")
    
    # Set priorities for each strategy type (lower is higher priority)
    priorities = {
        'moving_average_crossover': 1,  # Top priority
        'trend_following': 2,
        'volatility_breakout': 3,
        'mean_reversion': 4,
        'volume_spike': 5
    }
    
    # Add strategies in order of priority (using sort)
    strategy_order = sorted(strategy_names, key=lambda x: priorities.get(x, 100))
    
    for name in strategy_order:
        if name == 'trend_following':
            strategies.append(TrendFollowingStrategy(
                fast_period=fast_period,
                slow_period=slow_period,
                signal_threshold=0.0,
                volume_threshold=volume_threshold,
                priority=priorities.get('trend_following', 2)
            ))
        elif name == 'mean_reversion':
            strategies.append(MeanReversionStrategy(
                rsi_period=rsi_period,
                oversold_threshold=oversold_threshold,
                overbought_threshold=overbought_threshold
            ))
        elif name == 'moving_average_crossover':
            # Create MAC strategy with enhanced parameters
            strategies.append(MovingAverageCrossoverStrategy(
                fast_period=fast_period,
                slow_period=slow_period,
                signal_threshold=0.0,
                volume_threshold=volume_threshold,
                priority=priorities.get('moving_average_crossover', 1),
                rsi_period=rsi_period,
                rsi_overbought=overbought_threshold,
                rsi_oversold=oversold_threshold,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                atr_period=atr_period,
                adaptive_params=adaptive_params,
                use_market_regime=use_market_regime,
                use_symbol_specific_params=use_symbol_specific_params,
                param_db_path=param_db_path
            ))
            
        elif name == 'volatility_breakout':
            strategies.append(VolatilityBreakoutStrategy(
                lookback_period=20,
                volatility_threshold=volatility_threshold,
                volume_threshold=volume_threshold,
                consolidation_threshold=consolidation_threshold
            ))
        elif name == 'volume_spike':
            strategies.append(VolumeSpikeStrategy(
                lookback_period=20,
                volume_threshold=2.0,
                price_change_threshold=0.5,  # More conservative (was 0.7)
                min_consolidation_days=5
            ))
    return strategies

def initialize_components(symbol, start_date, end_date, initial_capital, strategies,
                          timeframe, fast_period=15, slow_period=50, rsi_period=14,
                          oversold_threshold=25, overbought_threshold=75, stop_loss=0.03, 
                          take_profit=0.06, position_size=0.05, slippage_model='percentage', slippage_value=0.001,
                          commission_model='percentage', commission_value=0.001,
                          market='custom', use_influxdb=True, 
                          macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14,
                          adaptive_params=True, use_market_regime=True,
                          use_symbol_specific_params=False, param_db_path="data/strategy_params.db"):
    """
    Initialize backtest components for a symbol.
    
    Args:
        symbol: Symbol to initialize components for
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for backtest
        strategies: List of strategy names to use
        timeframe: Timeframe for backtest
        fast_period: Fast period for trend following strategy
        slow_period: Slow period for trend following strategy
        rsi_period: RSI period for mean reversion strategy
        oversold_threshold: Oversold threshold for mean reversion strategy
        overbought_threshold: Overbought threshold for mean reversion strategy
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        position_size: Position size as fraction of portfolio
        slippage_model: Slippage model
        slippage_value: Slippage value
        commission_model: Commission model
        commission_value: Commission value
        market: Market to backtest
        use_influxdb: Whether to use InfluxDB as the data source
        macd_fast: Fast period for MACD
        macd_slow: Slow period for MACD
        macd_signal: Signal period for MACD
        atr_period: Period for ATR calculation
        adaptive_params: Whether to use adaptive parameters
        use_market_regime: Whether to use market regime detection
        use_symbol_specific_params: Whether to use symbol-specific parameters 
        param_db_path: Path to the parameter database
        
    Returns:
        Tuple of (data_provider, backtest_engine)
    """
    # Convert parameters to appropriate types
    initial_capital = Decimal(str(initial_capital))
    stop_loss = Decimal(str(stop_loss))
    take_profit = Decimal(str(take_profit))
    position_size = Decimal(str(position_size))
    slippage_value = Decimal(str(slippage_value))
    commission_value = Decimal(str(commission_value))
    
    # Handle strategies
    strategy_instances = []
    if strategies is None:
        strategies = ['trend_following', 'mean_reversion']
    
    for strategy_name in strategies:
        if strategy_name == 'trend_following':
            strategy_instances.append(TrendFollowingStrategy(
                fast_period=fast_period,
                slow_period=slow_period
            ))
        elif strategy_name == 'mean_reversion':
            strategy_instances.append(MeanReversionStrategy(
                rsi_period=rsi_period,
                oversold_threshold=oversold_threshold,
                overbought_threshold=overbought_threshold
            ))
        elif strategy_name == 'moving_average_crossover':
            strategy_instances.append(MovingAverageCrossoverStrategy(
                fast_period=fast_period,
                slow_period=slow_period,
                volume_threshold=1.5,
                priority=1,  # Highest priority
                rsi_period=rsi_period,
                rsi_overbought=overbought_threshold,
                rsi_oversold=oversold_threshold,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                atr_period=atr_period,
                adaptive_params=adaptive_params,
                use_market_regime=use_market_regime,
                use_symbol_specific_params=use_symbol_specific_params,
                param_db_path=param_db_path
            ))
        elif strategy_name == 'volatility_breakout':
            strategy_instances.append(VolatilityBreakoutStrategy(
                lookback_period=20,
                volatility_threshold=2.0,
                volume_threshold=2.0,
                consolidation_threshold=0.4
            ))
        elif strategy_name == 'volume_spike':
            strategy_instances.append(VolumeSpikeStrategy(
                lookback_period=20,
                volume_threshold=3.0,
                price_change_threshold=0.7,
                min_consolidation_days=5
            ))
    
    # Initialize data provider
    # For custom market, use data directory directly
    # For other markets, use market-specific directory
    base_path = 'data' if market == 'custom' else f'data/{market}'
    data_provider = DataProvider(base_path=base_path, use_influxdb=use_influxdb)
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(initial_capital=initial_capital)
    
    # Initialize execution handler
    execution_handler = ExecutionHandler(
        slippage_model=slippage_model,
        slippage_params={"percentage": slippage_value},
        commission_model=commission_model,
        commission_params={"percentage": commission_value, "min_commission": Decimal("1.0")}
    )
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        data_provider=data_provider,
        strategies=strategy_instances,
        portfolio_manager=portfolio_manager,
        execution_handler=execution_handler
    )
    
    return data_provider, backtest_engine

def generate_report_directory(symbol_file: str) -> str:
    """Generate a directory path for the report based on the symbol file."""
    # Create a directory under reports/backtest with today's date
    today = datetime.now().strftime("%Y%m%d")
    basename = os.path.basename(symbol_file).split('.')[0]
    
    # Create directory structure for backtest reports
    report_dir = f"reports/backtest/{today}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Create symbol-specific directory
    symbol_dir = f"{report_dir}/symbols"
    os.makedirs(symbol_dir, exist_ok=True)
    
    return report_dir

def generate_report_basename(symbol_file: str, start_date: datetime, end_date: datetime) -> str:
    """Generate report base name.
    
    Args:
        symbol_file: Path to the symbol file
        start_date: Start date of the backtest
        end_date: End date of the backtest
        
    Returns:
        Base name for the report files
    """
    symbol_base = Path(symbol_file).stem
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    return f"{symbol_base}_summary_{start_date_str}_{end_date_str}"

def generate_html_report(results, output_dir, filename):
    """Generate an HTML report from the backtest results."""
    try:
        html_content = """<!DOCTYPE html>
        <html>
        <head>
            <title>Market Backtest Summary</title>
            <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .summary { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .positive { color: green; }
        .negative { color: red; }
        .market-summary { margin-bottom: 30px; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Market Backtest Summary</h1>
            <div class="summary">
                <h2>Backtest Results</h2>
        <p>Total symbols processed: {total_symbols}</p>
            </div>
            
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Final Equity</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Total Trades</th>
                </tr>
        """
        
        # Add rows for each result
        for result in results:
            symbol = result.get('symbol', 'Unknown')
            final_equity = result.get('final_equity', 0)
            total_return = result.get('total_return', 0)
            sharpe_ratio = result.get('sharpe_ratio', 0)
            max_drawdown = result.get('max_drawdown', 0)
            win_rate = result.get('win_rate', 0)
            total_trades = result.get('total_trades', 0)
            
            # Format values
            final_equity_formatted = "${:,.2f}".format(final_equity)
            total_return_formatted = "{:.2f}%".format(total_return * 100)
            sharpe_ratio_formatted = "{:.2f}".format(sharpe_ratio)
            max_drawdown_formatted = "{:.2f}%".format(max_drawdown * 100)
            win_rate_formatted = "{:.2f}%".format(win_rate * 100)
            
            # Determine the class for coloring returns
            return_class = "positive" if total_return >= 0 else "negative"
            
            # Create a link to the symbol report if it exists
            symbol_report_path = f"symbols/{symbol}/{symbol}_report.html"
                
                # Add row to table
                html_content += f"""
                <tr>
            <td><a href="{symbol_report_path}" target="_blank">{symbol}</a></td>
            <td>{final_equity_formatted}</td>
            <td class="{return_class}">{total_return_formatted}</td>
            <td>{sharpe_ratio_formatted}</td>
            <td>{max_drawdown_formatted}</td>
            <td>{win_rate_formatted}</td>
                    <td>{total_trades}</td>
        </tr>"""
        
        # Close the HTML table and document
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Replace total symbols placeholder
        html_content = html_content.replace("{total_symbols}", str(len(results)))
        
        # Write to file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path
    except Exception as e:
        logger.error(f"Error generating HTML summary report: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_csv_report(results: List[Dict], output_path: str):
    """Generate CSV report.
    
    Args:
        results: List of backtest results
        output_path: Path to save the CSV report
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create DataFrame from results
        data = []
        for result in results:
            if "error" not in result:
                metrics = result.get("metrics", {})
                data.append({
                    "symbol": result.get("symbol", "Unknown"),
                    "final_equity": result.get("final_equity", 0),
                    "total_return": metrics.get("total_return", 0) * 100,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0) * 100,
                    "win_rate": metrics.get("win_rate", 0) * 100,
                    "total_trades": metrics.get("total_trades", 0),
                    "execution_time": result.get("execution_time", 0)
                })
        
        # Create DataFrame and save to CSV
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"CSV report generated: {output_path}")
        else:
            logger.warning("No valid data to save in CSV report")
            
    except Exception as e:
        logger.error(f"Error generating CSV report: {str(e)}")
        logger.error(traceback.format_exc())

def generate_json_report(results: List[Dict], output_path: str):
    """Generate JSON report.
    
    Args:
        results: List of backtest results
        output_path: Path to save the JSON report
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract summary data
        summary_data = {
            "total_symbols": len(results),
            "successful_symbols": sum(1 for r in results if "error" not in r),
            "symbols": []
        }
        
        # Add data for each symbol
        for result in results:
            if "error" not in result:
                metrics = result.get("metrics", {})
                summary_data["symbols"].append({
                    "symbol": result.get("symbol", "Unknown"),
                    "final_equity": float(result.get("final_equity", 0)),
                    "total_return": float(metrics.get("total_return", 0) * 100),
                    "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0) * 100),
                    "win_rate": float(metrics.get("win_rate", 0) * 100),
                    "total_trades": int(metrics.get("total_trades", 0)),
                    "execution_time": float(result.get("execution_time", 0))
                })
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
            
        logger.info(f"JSON report generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating JSON report: {str(e)}")
        logger.error(traceback.format_exc())

def get_symbol_data(symbol, data_source, timeframe, start_date, end_date, symbols_dir=None):
    """Get data for a symbol from the specified data source.
    
    Args:
        symbol: Symbol to get data for
        data_source: Data source ('csv', 'api', etc.)
        timeframe: Timeframe for the data
        start_date: Start date for the data
        end_date: End date for the data
        symbols_dir: Directory containing symbol data files
        
    Returns:
        DataFrame with symbol data
    """
    try:
        # Use pandas to read data from CSV
        if data_source.lower() == 'csv':
            # If symbols_dir is provided, use it
            base_dir = symbols_dir if symbols_dir else 'data/market'
            file_path = f"{base_dir}/{symbol}.csv"
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Data file for {symbol} not found at {file_path}")
                return None
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Ensure start_date and end_date are timezone-naive for consistent comparison
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            elif start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
                
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            elif end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)
            
            # Ensure the dataframe index is timezone-naive
            if df.index.tzinfo is not None:
                df.index = df.index.tz_localize(None)
                
            # Now filter by date with consistent timezone handling
            df = df.loc[start_date:end_date]
            
            # Check if we have data
            if len(df) == 0:
                logger.warning(f"No data for {symbol} in the specified date range")
                return None
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try lowercase column names
                lowercase_map = {col: col.lower() for col in missing_columns}
                for orig, lower in lowercase_map.items():
                    if lower in df.columns:
                        df[orig] = df[lower]
                        missing_columns.remove(orig)
            
            # If still missing columns, log warning and return None
            if missing_columns:
                logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
                return None
                
            return df
            
        # TODO: Implement other data sources as needed
        else:
            logger.error(f"Unsupported data source: {data_source}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def setup_logging(log_level='INFO'):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
    """
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Base configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def run_single_backtest(
    symbol: str,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    data_source: str = 'csv',
    timeframe: str = '1d',
    initial_capital: float = 100000,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    risk_free_rate: float = 0.02,
    strategy_params: Dict = None,
    generate_reports: bool = False,
    chart_intersections: bool = True,
    output_dir: str = 'reports/backtest',
    symbols_dir: str = None,
    additional_plots: List[str] = None,
    custom_params: Dict = None,
    **kwargs
) -> Dict:
    """Run backtest for a single symbol.
    
    Args:
        symbol: Symbol to backtest
        strategy_name: Name of strategy to use
        start_date: Start date for backtest
        end_date: End date for backtest
        data_source: Data source to use
        timeframe: Timeframe for data
        initial_capital: Initial capital
        commission_rate: Commission rate
        slippage_rate: Slippage rate
        risk_free_rate: Risk-free rate
        strategy_params: Strategy parameters
        generate_reports: Whether to generate reports
        chart_intersections: Whether to chart intersections
        output_dir: Output directory for reports
        symbols_dir: Directory containing symbol data
        additional_plots: Additional plots to generate
        custom_params: Custom parameters for strategy
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict containing backtest results
    """
    try:
        # Initialize strategy
        params = strategy_params or {}
        if custom_params:
            params.update(custom_params)
            
        if strategy_name.lower() == 'moving_average_crossover':
            strategy = MovingAverageCrossover(
                fast_period=params.get('fast_period', 10),
                slow_period=params.get('slow_period', 30),
                stop_loss=params.get('stop_loss', 0.02),
                take_profit=params.get('take_profit', 0.05),
                position_size=params.get('position_size', 0.1)
            )
        elif strategy_name.lower() == 'rsi_strategy':
            strategy = RSIStrategy(
                period=params.get('rsi_period', 14),
                oversold=params.get('oversold_threshold', 30),
                overbought=params.get('overbought_threshold', 70),
                stop_loss=params.get('stop_loss', 0.02),
                take_profit=params.get('take_profit', 0.05),
                position_size=params.get('position_size', 0.1)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Initialize data provider
        data_provider = DataProvider(base_path=symbols_dir if symbols_dir else 'data/market')
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(initial_capital=Decimal(str(initial_capital)))
        
        # Initialize execution handler
        execution_handler = ExecutionHandler(
            slippage_model='percentage',
            slippage_params={'percentage': Decimal(str(slippage_rate))},
            commission_model='percentage',
            commission_params={'percentage': Decimal(str(commission_rate))}
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=Decimal(str(initial_capital)),
            start_date=start_date,
            end_date=end_date,
            data_provider=data_provider,
            strategies=[strategy],
            portfolio_manager=portfolio_manager,
            execution_handler=execution_handler
        )
        
        # Run backtest
        results = engine.run(
            symbols=[symbol],
            timeframe=timeframe,
            risk_percentage=Decimal('0.01'),
            show_progress=False
        )
        
        # Generate reports if requested
        if generate_reports:
            generate_report(
                results=results,
                output_dir=output_dir,
                symbol=symbol,
                strategy_name=strategy_name,
                chart_intersections=chart_intersections,
                additional_plots=additional_plots
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'success': False,
            'error': str(e)
        }

def run_parallel_backtest(
    symbols, start_date, end_date, strategies, timeframe='1d',
    initial_capital=100000, slippage_rate=0.001, commission_rate=0.001,
    risk_free_rate=0.02, position_size=0.2, data_source='csv',
    generate_reports=False, chart_intersections=True, show_progress=True,
    num_workers=None, output_dir='reports/backtest', symbols_dir=None,
    additional_plots=None, custom_params=None, **kwargs
):
    """Run backtest for multiple symbols in parallel.
    
    Args:
        symbols: List of symbols to run backtest for
        strategies: List of strategies to run
        ...
        
    Returns:
        List of results for each symbol
    """
    start_time = time.time()
    
    # Set default number of workers based on CPU count
    if num_workers is None:
        cpu_count = os.cpu_count()
        num_workers = max(1, cpu_count - 1)  # Leave 1 CPU for system
    
    # Get symbols if a file path is provided
    if isinstance(symbols, str) and os.path.exists(symbols):
        symbols = load_symbols(symbols)
    
    # Parse dates and ensure they are timezone-naive
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    elif start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
        
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    elif end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
    
    # Initialize results
    results = []
    
    # Create single strategy instance or strategy for each symbol
    strategy_name = strategies[0] if isinstance(strategies, list) else strategies
    
    try:
        # Run in parallel
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit jobs
        for symbol in symbols:
                # Submit job with parameters
                future = executor.submit(
                    run_single_backtest,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    timeframe=timeframe,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_rate=slippage_rate,
                    risk_free_rate=risk_free_rate,
                    strategy_params=kwargs.get('strategy_params'),
                    generate_reports=generate_reports,
                    chart_intersections=chart_intersections,
                    output_dir=output_dir,
                    symbols_dir=symbols_dir,
                    additional_plots=additional_plots,
                    custom_params=custom_params
                )
                futures.append((symbol, future))
            
            # Progress bar
            if show_progress and len(symbols) > 1:
                futures_iterator = tqdm(futures, desc="Running backtests", unit="symbol")
        else:
                futures_iterator = futures
            
            # Gather results
            for symbol, future in futures_iterator:
                try:
                    # Get result
                    metrics = future.result()
                    
                    # Store result
                    if metrics:
                        results.append(metrics)
                    else:
                        logger.warning(f"Backtest failed for {symbol}")
                        results.append({
                            'symbol': symbol,
                            'success': False,
                            'error': 'Unknown error'
                        })
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    logger.error(traceback.format_exc())
                    results.append({
                        'symbol': symbol,
                        'success': False,
                        'error': str(e)
                    })
    
    except Exception as e:
        logger.error(f"Error in parallel backtest: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Calculate end time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Backtest completed in {execution_time:.2f} seconds")
    
    return results

def generate_summary_report(results, output_dir, start_date_str, end_date_str, symbol_file=None):
    """Generate a summary report from backtest results."""
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the timestamp for filename
    filename_base = f"{start_date_str}_{end_date_str}"
    if symbol_file:
        filename_base = f"{os.path.basename(symbol_file).split('.')[0]}_{filename_base}"
    
    # Create CSV summary
    try:
        logger.info(f"Generating summary report for {len(results)} results")
        
        # Filter out results without the required data
        valid_results = [r for r in results if r.get('symbol') is not None and 'final_equity' in r]
        logger.info(f"Found {len(valid_results)} valid results out of {len(results)} total results")
        
        # Format the results for output
        formatted_results = []
        for r in valid_results:
            # Extract metrics
            formatted_result = {
                'symbol': r.get('symbol'),
                'final_equity': r.get('final_equity', 0),
                'total_return': r.get('total_return', 0),
                'sharpe_ratio': r.get('sharpe_ratio', 0),
                'max_drawdown': r.get('max_drawdown', 0),
                'win_rate': r.get('win_rate', 0),
                'total_trades': r.get('total_trades', 0),
                'execution_time': r.get('execution_time', 0)
            }
            formatted_results.append(formatted_result)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(formatted_results)
        csv_path = os.path.join(output_dir, f"symbols_summary_{filename_base}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to CSV: {csv_path}")
        
        # Save to JSON
        json_path = os.path.join(output_dir, f"symbols_summary_{filename_base}.json")
        with open(json_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        logger.info(f"JSON report generated: {json_path}")
        
        # Generate HTML report
        try:
            generate_html_report(formatted_results, output_dir, f"symbols_summary_{filename_base}.html")
            logger.info(f"Summary report generated at {os.path.join(output_dir, f'symbols_summary_{filename_base}.html')}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            logger.error(traceback.format_exc())
        
        return csv_path
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def print_system_info():
    """Print system information for debugging."""
    logger.info(f"CPU Count: {multiprocessing.cpu_count()}")
    print(f"CPU Count: {multiprocessing.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Python Version: {'.'.join(map(str, tuple(sys.version_info)[:3]))}")
    print(f"Python Version: {'.'.join(map(str, tuple(sys.version_info)[:3]))}")

def load_symbols(symbol_source):
    """Load symbols from a source.
    
    Args:
        symbol_source: Source of symbols (file path, comma-separated string, etc.)
        
    Returns:
        List of symbols
    """
    # Check if the source is a string with commas
    if isinstance(symbol_source, str) and ',' in symbol_source:
        return [s.strip() for s in symbol_source.split(',')]
    
    # Check if the source is a file path
    if isinstance(symbol_source, str) and os.path.exists(symbol_source):
        # Check file extension
        ext = os.path.splitext(symbol_source)[1].lower()
        
        if ext == '.json':
            # Load from JSON file
            try:
                with open(symbol_source, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # Direct list of symbols
                    return data
                elif isinstance(data, dict) and 'symbols' in data:
                    # Dictionary with 'symbols' key
                    return data['symbols']
                elif isinstance(data, dict):
                    # Dictionary with symbol keys
                    return list(data.keys())
    else:
                    logger.warning(f"Unknown JSON format in {symbol_source}")
                    return []
            except Exception as e:
                logger.error(f"Error loading JSON file {symbol_source}: {str(e)}")
                return []
        
        elif ext == '.csv':
            # Load from CSV file
            try:
                df = pd.read_csv(symbol_source)
                
                # Try different column names for symbols
                for col_name in ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER']:
                    if col_name in df.columns:
                        return df[col_name].tolist()
                
                # If no known column found, use the first column
                return df.iloc[:, 0].tolist()
            except Exception as e:
                logger.error(f"Error loading CSV file {symbol_source}: {str(e)}")
                return []
        
        elif ext == '.txt':
            # Load from text file (one symbol per line)
            try:
                with open(symbol_source, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"Error loading text file {symbol_source}: {str(e)}")
                return []
        
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return []
    
    # If the source is already a list, return it
    if isinstance(symbol_source, list):
        return symbol_source
    
    # If it's a single symbol string
    if isinstance(symbol_source, str):
        return [symbol_source]
    
    # Unknown format
    logger.warning(f"Unknown symbol source format: {type(symbol_source)}")
    return []

def run_market_backtest(
    symbols, strategies, start_date, end_date, timeframe='1d',
    execution_mode='sequential', workers=None, initial_capital=100000,
    slippage_value=0.001, commission_value=0.001, position_size=0.2,
    output_dir='reports/backtest', generate_reports=False, 
    show_progress=False, custom_params=None, **kwargs
):
    """Run a market backtest.
    
    Args:
        symbols: List of symbols or path to symbol file
        strategies: List of strategies or single strategy name
        start_date: Start date
        end_date: End date
        timeframe: Timeframe
        execution_mode: Execution mode ('sequential', 'parallel', 'async')
        workers: Number of workers for parallel execution
        initial_capital: Initial capital
        slippage_value: Slippage value
        commission_value: Commission value
        position_size: Position size
        output_dir: Directory to save reports
        generate_reports: Whether to generate reports
        show_progress: Whether to show progress bar
        custom_params: Custom parameters for strategies
        
    Returns:
        List of backtest results
    """
    # Handle different execution modes
    if execution_mode == 'parallel':
        return run_parallel_backtest(
            symbols=symbols,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_capital=initial_capital,
            slippage_rate=slippage_value,
            commission_rate=commission_value,
            position_size=position_size,
            data_source='csv',
            generate_reports=generate_reports,
            show_progress=show_progress,
            num_workers=workers,
            output_dir=output_dir,
            custom_params=custom_params,
            **kwargs
        )
    elif execution_mode == 'async':
        logger.warning("Async execution mode is not fully implemented yet, falling back to parallel")
        return run_parallel_backtest(
            symbols=symbols,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_capital=initial_capital,
            slippage_rate=slippage_value,
            commission_rate=commission_value,
            position_size=position_size,
            data_source='csv',
            generate_reports=generate_reports,
            show_progress=show_progress,
            num_workers=workers,
            output_dir=output_dir,
            custom_params=custom_params,
            **kwargs
        )
    else:
        # Sequential execution
        results = []
        
        # Run backtest for each symbol
        iterator = tqdm(symbols, desc="Running backtests") if show_progress else symbols
        for symbol in iterator:
            try:
                # Default to first strategy if multiple provided
                strategy_name = strategies[0] if isinstance(strategies, list) else strategies
                
                # Run backtest
                result = run_single_backtest(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    start_date=start_date,
                    end_date=end_date,
                    data_source='csv',
                    timeframe=timeframe,
                    initial_capital=initial_capital,
                    commission_rate=commission_value,
                    slippage_rate=slippage_value,
                    generate_reports=generate_reports,
                    output_dir=output_dir,
                    custom_params=custom_params,
                    symbols_dir=symbols_dir
                )
                
                # Store result
                results.append(result)
            except Exception as e:
                logger.error(f"Error running backtest for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })
    
    return results

def initialize_strategy(strategy_name, strategy_params=None, custom_params=None, symbol=None):
    """Initialize a strategy with parameters.
    
    Args:
        strategy_name: Name of the strategy to initialize
        strategy_params: Dictionary of strategy parameters
        custom_params: Dictionary of custom parameters
        symbol: Symbol to use for the strategy
        
    Returns:
        Initialized strategy instance
    """
    # Get parameters based on hierarchy (strategy_params > custom_params > defaults)
    params = {}
    
    # Use custom_params as base if provided
    if custom_params:
        params.update(custom_params)
    
    # Override with strategy_params if provided
    if strategy_params:
        params.update(strategy_params)
    
    # Check for strategy type
    if strategy_name.lower() == 'moving_average_crossover':
        from src.market_system.strategies.moving_average_crossover import MovingAverageCrossover
        return MovingAverageCrossover(
            fast_period=params.get('fast_period', 12),
            slow_period=params.get('slow_period', 26),
            stop_loss=params.get('stop_loss', 0.02),
            take_profit=params.get('take_profit', 0.05),
            position_size=params.get('position_size', 0.1)
    elif strategy_name.lower() == 'rsi_strategy':
        from src.market_system.strategies.rsi_strategy import RSIStrategy
        return RSIStrategy(
            rsi_period=params.get('rsi_period', 14),
            oversold_threshold=params.get('oversold_threshold', 30),
            overbought_threshold=params.get('overbought_threshold', 70),
            stop_loss=params.get('stop_loss', 0.02),
            take_profit=params.get('take_profit', 0.05),
            position_size=params.get('position_size', 0.1)
    elif strategy_name.lower() == 'macd_strategy':
        from src.market_system.strategies.macd_strategy import MACDStrategy
        return MACDStrategy(
            fast_period=params.get('macd_fast', 12),
            slow_period=params.get('macd_slow', 26),
            signal_period=params.get('macd_signal', 9),
            stop_loss=params.get('stop_loss', 0.02),
            take_profit=params.get('take_profit', 0.05),
            position_size=params.get('position_size', 0.1)
        )
        else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def main():
    """Main entry point for market backtest."""
    parser = argparse.ArgumentParser(description='Run market backtest')
    parser.add_argument('--symbol', type=str, help='Symbol to backtest')
    parser.add_argument('--symbol-file', type=str, help='File containing symbols to backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategies', type=str, help='Strategy to use (comma-separated list)')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe for data')
    parser.add_argument('--execution-mode', type=str, default='sequential', help='Execution mode (sequential, parallel, async)')
    parser.add_argument('--workers', type=int, help='Number of workers for parallel execution')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--slippage-value', type=float, default=0.001, help='Slippage rate')
    parser.add_argument('--commission-value', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--position-size', type=float, default=0.2, help='Position size')
    parser.add_argument('--generate-reports', action='store_true', help='Generate reports')
    parser.add_argument('--show-progress', action='store_true', help='Show progress bar')
    parser.add_argument('--custom-params', type=str, help='Custom parameters for strategy')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Load symbols
    symbols = []
    if args.symbol:
        symbols = args.symbol.split(',')
    elif args.symbol_file:
        symbols = load_symbols(args.symbol_file)
            else:
        logger.error("No symbols provided")
            return
            
    logger.info(f"Loaded {len(symbols)} symbols from command line:\n{', '.join(symbols)}")
    print(f"Loaded {len(symbols)} symbols from command line:\n{', '.join(symbols)}")
    
    # Parse dates and ensure they are timezone-naive
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        # Default to 1 year ago if no start date provided
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
        start_date = end_date - timedelta(days=365)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Parse custom parameters if provided
    custom_params = {}
    if args.custom_params:
        try:
            custom_params = json.loads(args.custom_params)
        except json.JSONDecodeError:
            logger.error("Invalid custom parameters JSON")
            return
    
    # Find symbols directory
    symbols_dir = None
    if os.path.exists('data/stock'):
        symbols_dir = 'data/stock'
    elif os.path.exists('data/market'):
        symbols_dir = 'data/market'
        else:
        # Try to find any directory containing CSV files
        for root, dirs, files in os.walk('data'):
            if any(f.endswith('.csv') for f in files):
                symbols_dir = root
                break
                
    logger.info(f"Using symbols directory: {symbols_dir}")
    print(f"Using symbols directory: {symbols_dir}")
    
    # Run backtest
    reports_dir = f"reports/backtest/{datetime.now().strftime('%Y%m%d')}"
    results = []
    
    if args.execution_mode in ['parallel', 'async']:
        # Run in parallel or async mode
        results = run_market_backtest(
            symbols=symbols,
            strategies=args.strategies,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe,
            execution_mode=args.execution_mode,
            workers=args.workers,
            initial_capital=args.initial_capital,
            slippage_value=args.slippage_value,
            commission_value=args.commission_value,
            position_size=args.position_size,
            output_dir=reports_dir,
            generate_reports=args.generate_reports,
            show_progress=args.show_progress,
            custom_params=custom_params,
            symbols_dir=symbols_dir
        )
    else:
        # Run in sequential mode
        for symbol in tqdm(symbols, desc="Running backtests") if args.show_progress else symbols:
            try:
                metrics = run_single_backtest(
                    symbol=symbol,
                    strategy_name=args.strategies[0] if isinstance(args.strategies, list) else args.strategies,
                    start_date=start_date,
                    end_date=end_date,
                    data_source='csv',
                    timeframe=args.timeframe,
                    initial_capital=args.initial_capital,
                    commission_rate=args.commission_value,
                    slippage_rate=args.slippage_value,
                    generate_reports=args.generate_reports,
                    output_dir=reports_dir,
                    custom_params=custom_params,
                    symbols_dir=symbols_dir
                )
                results.append(metrics)
    except Exception as e:
                logger.error(f"Error running backtest for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })
    
    # Generate summary report
    if args.generate_reports and results:
        try:
            if isinstance(start_date, str):
                start_date_str = start_date.replace('-', '')
        else:
                start_date_str = start_date.strftime('%Y%m%d')
                
            if isinstance(end_date, str):
                end_date_str = end_date.replace('-', '')
            else:
                end_date_str = end_date.strftime('%Y%m%d')
            
            # Generate summary report
            generate_summary_report(
                results=results,
                output_dir=reports_dir,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                symbol_file=args.symbol_file
            )
            logger.info(f"Results stored in {reports_dir}")
            print(f"Results stored in {reports_dir}")
    except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
        logger.error(traceback.format_exc())
        
    # Calculate execution time
    end_time = time.time()
    logger.info(f"Backtest completed in {end_time - start_time:.2f} seconds")
    print(f"Backtest completed in {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main() 
else:
    # Make run_single_backtest available at module level for multiprocessing
    __all__ = ['run_single_backtest', 'run_parallel_backtest', 'run_market_backtest', 
              'run_market_backtest_async', 'run_symbol_backtest_async', 
              'initialize_components', 'initialize_strategies'] 
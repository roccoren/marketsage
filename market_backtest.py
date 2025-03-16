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
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import pandas as pd
import concurrent.futures
import argparse
import multiprocessing
from tqdm import tqdm

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

from src.market_system.backtesting.backtest_engine import BacktestEngine
from src.market_system.backtesting.data_provider import DataProvider
from src.market_system.backtesting.portfolio_manager import PortfolioManager
from src.market_system.backtesting.execution_handler import ExecutionHandler
from src.market_system.backtesting.strategies import (
    TrendFollowingStrategy, MeanReversionStrategy,
    MovingAverageCrossoverStrategy, VolatilityBreakoutStrategy, VolumeSpikeStrategy
)
from src.market_system.backtesting.report_generator import ReportGenerator

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

def get_market_symbols(data_dir: Path, market: str = "hkex") -> list:
    """Get all available symbols from the specified market data directory."""
    market_dir = data_dir / market
    if not market_dir.exists():
        logger.error(f"{market.upper()} data directory not found: {market_dir}")
        return []
    
    symbols = [d.name for d in market_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(symbols)} {market.upper()} symbols")
    return symbols

def initialize_components(symbol, start_date, end_date, initial_capital=10000, 
                          strategies=None, timeframe='1d', fast_period=20, slow_period=50,
                          rsi_period=14, oversold_threshold=30, overbought_threshold=70,
                          stop_loss=0.05, take_profit=0.1, position_size=0.1,
                          slippage_model='percentage', slippage_value=0.001,
                          commission_model='percentage', commission_value=0.001,
                          market='hkex'):
    """
    Initialize backtest components for a symbol
    
    Args:
        symbol: Symbol to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for backtest
        strategies: List of strategy instances to use
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
        
    Returns:
        Tuple of (data_provider, backtest_engine)
    """
    # Initialize data provider
    data_provider = DataProvider(base_path='data')
    
    # Initialize strategies if not provided
    if strategies is None:
        # Default to trend following strategy
        trend_strategy = TrendFollowingStrategy(
            fast_period=fast_period,
            slow_period=slow_period
        )
        strategies = [trend_strategy]
        
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(initial_capital=initial_capital)
    
    # Initialize execution handler
    execution_handler = ExecutionHandler()
    
    # Convert start_date and end_date to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategies=strategies,
        portfolio_manager=portfolio_manager,
        execution_handler=execution_handler,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        data_provider=data_provider
    )
    
    return data_provider, backtest_engine

def run_backtest_for_symbol(symbol, start_date, end_date, timeframe='1d', 
                           initial_capital=100000, strategies=None, 
                           data_provider=None, generate_reports=True,
                           fast_period=20, slow_period=50,
                           rsi_period=14, oversold_threshold=30, overbought_threshold=70,
                           stop_loss=0.05, take_profit=0.1, position_size=0.1,
                           slippage_model='percentage', slippage_value=0.001,
                           commission_model='percentage', commission_value=0.001,
                           market='hkex'):
    """
    Run backtest for a single symbol
    
    Args:
        symbol: Symbol to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Timeframe for backtest
        initial_capital: Initial capital for backtest
        strategies: List of strategy instances to use
        data_provider: Data provider instance (optional)
        generate_reports: Whether to generate reports
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
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Initialize components
        if data_provider is None:
            data_provider, backtest_engine = initialize_components(
                symbol, start_date, end_date, initial_capital, strategies, timeframe,
                fast_period, slow_period, rsi_period, oversold_threshold, overbought_threshold,
                stop_loss, take_profit, position_size, slippage_model, slippage_value,
                commission_model, commission_value, market
            )
        else:
            _, backtest_engine = initialize_components(
                symbol, start_date, end_date, initial_capital, strategies, timeframe,
                fast_period, slow_period, rsi_period, oversold_threshold, overbought_threshold,
                stop_loss, take_profit, position_size, slippage_model, slippage_value,
                commission_model, commission_value, market
            )
        
        # Run backtest
        start_time = time.time()
        print(f"Starting backtest for 1 symbols with timeframe {timeframe}")
        
        try:
            backtest_results = backtest_engine.run([symbol], data_provider, timeframe, market=market)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            print(f"Backtest completed in {execution_time:.2f} seconds")
            
            # Print summary
            final_equity = backtest_results['portfolio_value'][-1]
            total_return = (final_equity - initial_capital) / initial_capital
            print(f"Final equity: {final_equity:.2f} (Return: {total_return:.2%})")
            
            # Generate reports
            if generate_reports:
                # Create report generator
                report_dir = f"reports/backtest/{market}_market"
                report_generator = ReportGenerator(output_dir=report_dir)
                
                # Generate report name
                symbol_name = symbol.replace('.', '_')
                report_name = f"{market}_{symbol_name}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
                
                # Generate reports
                report_generator.generate_report(
                    backtest_results=backtest_results,
                    report_name=report_name,
                    formats=['html', 'json', 'csv']
                )
                
                # Generate charts
                chart_dir = Path(f"{report_dir}/{report_name}/charts")
                chart_dir.mkdir(exist_ok=True, parents=True)
                report_generator.generate_charts(
                    backtest_results=backtest_results,
                    chart_dir=chart_dir
                )
                
                # Print report path
                report_path = f"{market}_{symbol_name}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.html"
                print(f"Generated reports for {report_name} in formats: html, json, csv")
                print(f"Generated report in formats: html, json, csv")
            
            # Return results
            return {
                'symbol': symbol,
                'total_return': total_return,
                'sharpe_ratio': backtest_results.get('metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('metrics', {}).get('max_drawdown', 0),
                'win_rate': backtest_results.get('metrics', {}).get('win_rate', 0),
                'total_trades': len(backtest_results.get('trades', [])),
                'final_equity': final_equity,
                'report_path': report_path if generate_reports else ""
            }
        except Exception as e:
            print(f"Error in backtest execution: {str(e)}")
            # Return default results
            return {
                'symbol': symbol,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'final_equity': initial_capital,
                'report_path': "",
                'error': str(e)
            }
    except Exception as e:
        print(f"Error running backtest for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'final_equity': initial_capital,
            'report_path': "",
            'error': str(e)
        }

def run_market_backtest_parallel(symbols, start_date, end_date, timeframe='1d', 
                                workers=None, show_progress=False, 
                                generate_reports=False, data_provider=None,
                                strategies=None, fast_period=20, slow_period=50,
                                rsi_period=14, oversold_threshold=30, overbought_threshold=70,
                                stop_loss=0.05, take_profit=0.1, position_size=0.1,
                                slippage_model='percentage', slippage_value=0.001,
                                commission_model='percentage', commission_value=0.001,
                                initial_capital=100000, market='hkex'):
    """
    Run backtest for multiple symbols in parallel
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Timeframe for backtest
        workers: Number of worker processes
        show_progress: Whether to show progress bar
        generate_reports: Whether to generate individual reports
        data_provider: Shared data provider instance
        strategies: List of strategy instances to use
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
        initial_capital: Initial capital for backtest
        market: Market to backtest
        
    Returns:
        List of backtest results
    """
    # Set number of workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    # Create progress bar if requested
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(symbols), desc="Backtesting")
    
    # Initialize results
    results = []
    
    # Create strategy instances for each worker
    worker_strategies = []
    if strategies is not None:
        for strategy_name in strategies:
            if strategy_name == 'trend_following':
                worker_strategies.append(TrendFollowingStrategy(
                    fast_period=fast_period,
                    slow_period=slow_period,
                ))
            elif strategy_name == 'mean_reversion':
                worker_strategies.append(MeanReversionStrategy(
                    rsi_period=rsi_period,
                    oversold_threshold=oversold_threshold,
                    overbought_threshold=overbought_threshold,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                ))
            elif strategy_name == 'moving_average_crossover':
                worker_strategies.append(MovingAverageCrossoverStrategy(
                    fast_period=fast_period,
                    slow_period=slow_period,
                    volume_threshold=1.5,
                    vwap_filter=True
                ))
            elif strategy_name == 'volatility_breakout':
                worker_strategies.append(VolatilityBreakoutStrategy(
                    lookback_period=20,
                    volatility_threshold=1.5,
                    volume_threshold=1.5,
                    consolidation_threshold=0.5
                ))
            elif strategy_name == 'volume_spike':
                worker_strategies.append(VolumeSpikeStrategy(
                    lookback_period=20,
                    volume_threshold=2.5,
                    price_change_threshold=0.5,
                    min_consolidation_days=3
                ))
    else:
        # Default to trend following strategy
        worker_strategies.append(TrendFollowingStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        ))
    
    # Create shared data provider if requested
    shared_data_provider = None
    if data_provider is True:  # If share_data_provider is True
        shared_data_provider = DataProvider(base_path='data')
    
    # Run backtests in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(
                run_backtest_for_symbol, 
                symbol, 
                start_date, 
                end_date, 
                timeframe, 
                initial_capital,
                worker_strategies,
                shared_data_provider,
                generate_reports,
                fast_period, slow_period,
                rsi_period, oversold_threshold, overbought_threshold,
                stop_loss, take_profit, position_size,
                slippage_model, slippage_value,
                commission_model, commission_value,
                market
            ): symbol for symbol in symbols
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error running backtest for {symbol}: {str(e)}")
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
            
            # Update progress bar
            if pbar:
                pbar.update(1)
    
    # Close progress bar
    if pbar:
        pbar.close()
    
    return results

async def run_market_backtest_async(symbols, start_date, end_date, timeframe='1d', 
                                  max_concurrent=None, show_progress=False, 
                                  generate_reports=False, data_provider=None,
                                  strategies=None, fast_period=20, slow_period=50,
                                  rsi_period=14, oversold_threshold=30, overbought_threshold=70,
                                  stop_loss=0.05, take_profit=0.1, position_size=0.1,
                                  slippage_model='percentage', slippage_value=0.001,
                                  commission_model='percentage', commission_value=0.001,
                                  initial_capital=100000, market='hkex'):
    """
    Run backtest for multiple symbols asynchronously
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Timeframe for backtest
        max_concurrent: Maximum concurrent tasks
        show_progress: Whether to show progress bar
        generate_reports: Whether to generate individual reports
        data_provider: Shared data provider instance
        strategies: List of strategy instances to use
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
        initial_capital: Initial capital for backtest
        market: Market to backtest
        
    Returns:
        List of backtest results
    """
    # Set maximum concurrent tasks
    if max_concurrent is None:
        max_concurrent = multiprocessing.cpu_count() * 2
    
    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create progress bar if requested
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(symbols), desc="Backtesting")
    
    # Initialize results
    results = []
    
    # Create strategy instances
    worker_strategies = []
    if strategies is not None:
        for strategy_name in strategies:
            if strategy_name == 'trend_following':
                worker_strategies.append(TrendFollowingStrategy(
                    fast_period=fast_period,
                    slow_period=slow_period,
                ))
            elif strategy_name == 'mean_reversion':
                worker_strategies.append(MeanReversionStrategy(
                    rsi_period=rsi_period,
                    oversold_threshold=oversold_threshold,
                    overbought_threshold=overbought_threshold,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                ))
            elif strategy_name == 'moving_average_crossover':
                worker_strategies.append(MovingAverageCrossoverStrategy(
                    fast_period=fast_period,
                    slow_period=slow_period,
                    volume_threshold=1.5,
                    vwap_filter=True
                ))
            elif strategy_name == 'volatility_breakout':
                worker_strategies.append(VolatilityBreakoutStrategy(
                    lookback_period=20,
                    volatility_threshold=1.5,
                    volume_threshold=1.5,
                    consolidation_threshold=0.5
                ))
            elif strategy_name == 'volume_spike':
                worker_strategies.append(VolumeSpikeStrategy(
                    lookback_period=20,
                    volume_threshold=2.5,
                    price_change_threshold=0.5,
                    min_consolidation_days=3
                ))
    else:
        # Default to trend following strategy
        worker_strategies.append(TrendFollowingStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        ))
    
    # Define async backtest function
    async def run_backtest_async(symbol):
        """Run backtest for a single symbol asynchronously."""
        async with semaphore:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    None, 
                    run_backtest_for_symbol,
                    symbol, start_date, end_date, timeframe, 
                    initial_capital, worker_strategies, data_provider, generate_reports,
                    fast_period, slow_period, rsi_period, oversold_threshold, overbought_threshold,
                    stop_loss, take_profit, position_size, slippage_model, slippage_value,
                    commission_model, commission_value, market
                )
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                
                return result
            except Exception as e:
                logger.error(f"Error in backtest for {symbol}: {str(e)}")
                return None
    
    # Run backtests asynchronously
    start_time = time.time()
    tasks = [run_backtest_async(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    
    # Close progress bar
    if pbar is not None:
        pbar.close()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"Completed all backtests in {execution_time:.2f} seconds")
    
    # Generate summary report
    generate_summary_report(results, start_date, end_date, market=market)
    
    return results

def run_market_backtest(symbols, start_date, end_date, timeframe='1d', 
                       execution_mode='parallel', workers=None, max_concurrent=None,
                       show_progress=False, generate_reports=False, 
                       share_data_provider=False, strategies=None,
                       fast_period=20, slow_period=50,
                       rsi_period=14, oversold_threshold=30, overbought_threshold=70,
                       stop_loss=0.05, take_profit=0.1, position_size=0.1,
                       slippage_model='percentage', slippage_value=0.001,
                       commission_model='percentage', commission_value=0.001,
                       initial_capital=100000, market='hkex'):
    """
    Run market-wide backtest
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Timeframe for backtest
        execution_mode: Execution mode (parallel or async)
        workers: Number of worker processes for parallel execution
        max_concurrent: Maximum concurrent tasks for async execution
        show_progress: Whether to show progress bar
        generate_reports: Whether to generate individual reports
        share_data_provider: Whether to share data provider across workers
        strategies: List of strategy names to use
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
        initial_capital: Initial capital for backtest
        market: Market to backtest
        
    Returns:
        List of backtest results
    """
    # Create data provider if sharing
    data_provider = None
    if share_data_provider:
        data_provider = True  # Flag to create shared data provider
    
    # Run backtest based on execution mode
    start_time = time.time()
    if execution_mode == 'parallel':
        results = run_market_backtest_parallel(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            workers=workers,
            show_progress=show_progress,
            generate_reports=generate_reports,
            data_provider=data_provider,
            strategies=strategies,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            slippage_model=slippage_model,
            slippage_value=slippage_value,
            commission_model=commission_model,
            commission_value=commission_value,
            initial_capital=initial_capital,
            market=market
        )
    elif execution_mode == 'async':
        results = asyncio.run(run_market_backtest_async(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            max_concurrent=max_concurrent,
            show_progress=show_progress,
            generate_reports=generate_reports,
            data_provider=data_provider,
            strategies=strategies,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            slippage_model=slippage_model,
            slippage_value=slippage_value,
            commission_model=commission_model,
            commission_value=commission_value,
            initial_capital=initial_capital,
            market=market
        ))
    else:
        raise ValueError(f"Invalid execution mode: {execution_mode}")
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"Completed all backtests in {execution_time:.2f} seconds")
    
    # Generate summary report
    generate_summary_report(results, start_date, end_date, market=market)
    
    return results

def generate_summary_report(results, start_date, end_date, market='hkex'):
    """
    Generate summary report for all backtests
    
    Args:
        results: List of backtest results
        start_date: Start date for backtest
        end_date: End date for backtest
        market: Market being backtested
    """
    # Create summary directory
    summary_dir = Path(f"reports/backtest/{market}_market")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Format dates for filename
    start_date_str = start_date.replace('-', '')
    end_date_str = end_date.replace('-', '')
    
    # Create summary path
    summary_path = Path(f"reports/backtest/{market}_market/{market}_summary_{start_date_str}_{end_date_str}.csv")
    
    # Filter out results with errors that would make them invalid
    valid_results = []
    for r in results:
        # Only filter out results if they have an error AND don't have required fields
        if "error" in r and (
            'symbol' not in r or 
            'total_return' not in r or 
            'sharpe_ratio' not in r or 
            'max_drawdown' not in r
        ):
            logger.warning(f"Filtering out invalid result for symbol {r.get('symbol', 'unknown')}: {r.get('error', 'unknown error')}")
        else:
            valid_results.append(r)
    
    logger.info(f"Found {len(valid_results)} valid results out of {len(results)} total results")
    
    # Create DataFrame from results
    df = pd.DataFrame(valid_results)
    
    # Sort by total return
    if not df.empty and 'total_return' in df.columns:
        df = df.sort_values('total_return', ascending=False)
    
    # Save to CSV
    df.to_csv(summary_path, index=False)
    
    # Generate HTML summary
    html_path = Path(f"reports/backtest/{market}_market/{market}_summary_{start_date_str}_{end_date_str}.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{market.upper()} Market Backtest Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>{market.upper()} Market Backtest Summary</h1>
        <p>Period: {start_date} to {end_date}</p>
        <p>Total symbols: {len(valid_results)}</p>
        
    """
    
    if not df.empty:
        # Add performance statistics
        profitable_symbols = len(df[df['total_return'] > 0])
        avg_return = df['total_return'].mean()
        
        html_content += f"""
        <h2>Performance Summary</h2>
        <div class="summary">
            <ul>
                <li>Profitable Symbols: {profitable_symbols} ({profitable_symbols/len(df)*100:.1f}%)</li>
                <li>Average Return: {avg_return*100:.2f}%</li>
            </ul>
        </div>
        
        <h2>Symbol Results</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Total Trades</th>
                <th>Final Equity</th>
                <th>Report</th>
            </tr>
        """
        
        for _, row in df.iterrows():
            color_class = "positive" if row['total_return'] > 0 else "negative"
            # Create proper report link
            if 'report_path' in row and row['report_path']:
                report_link = f"<a href='{row['report_path']}'>View</a>"
            else:
                report_link = ""
            
            html_content += f"""
            <tr>
                <td>{row['symbol']}</td>
                <td class="{color_class}">{row['total_return']*100:.2f}%</td>
                <td>{row['sharpe_ratio']:.2f}</td>
                <td>{row['max_drawdown']*100:.2f}%</td>
                <td>{row['win_rate']*100:.1f}%</td>
                <td>{row['total_trades']}</td>
                <td>${row['final_equity']:.2f}</td>
                <td>{report_link}</td>
            </tr>
            """
        
        html_content += "</table>"
    else:
        html_content += "<p>No valid results found.</p>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Summary report generated at {html_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run market-wide backtest for multiple markets')
    
    # Symbol selection
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest (e.g. 0700.HK 9988.HK)')
    parser.add_argument('--symbol-file', help='File containing symbols to backtest')
    parser.add_argument('--all-symbols', action='store_true', help='Backtest all available symbols')
    parser.add_argument('--market', default='hkex', choices=['hkex', 'nasdaq', 'nyse', 'asx'], 
                        help='Market to backtest (e.g. hkex, nasdaq, nyse, asx)')
    
    # Date range
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe for backtest (e.g. 1d, 1h)')
    
    # Strategy parameters
    parser.add_argument('--strategies', nargs='+', default=['trend_following'], 
                        choices=['trend_following', 'mean_reversion', 'moving_average_crossover', 
                                'volatility_breakout', 'volume_spike'], 
                        help='Strategies to use for backtesting')
    parser.add_argument('--fast-period', type=int, default=2, help='Fast period for trend following')
    parser.add_argument('--slow-period', type=int, default=4, help='Slow period for trend following')
    parser.add_argument('--rsi-period', type=int, default=3, help='RSI period for mean reversion')
    parser.add_argument('--oversold-threshold', type=float, default=30, help='Oversold threshold for mean reversion')
    parser.add_argument('--overbought-threshold', type=float, default=70, help='Overbought threshold for mean reversion')
    
    # Position sizing and risk management
    parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.1, help='Position size as fraction of portfolio')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=0.1, help='Take profit percentage')
    
    # Execution parameters
    parser.add_argument('--slippage-model', default='percentage', choices=['fixed', 'percentage'], help='Slippage model')
    parser.add_argument('--slippage-value', type=float, default=0.001, help='Slippage value')
    parser.add_argument('--commission-model', default='percentage', choices=['fixed', 'percentage'], help='Commission model')
    parser.add_argument('--commission-value', type=float, default=0.001, help='Commission value')
    
    # Execution mode
    parser.add_argument('--execution-mode', default='parallel', choices=['parallel', 'async'], help='Execution mode')
    parser.add_argument('--workers', type=int, help='Number of worker processes for parallel execution')
    parser.add_argument('--max-concurrent', type=int, help='Maximum concurrent tasks for async execution')
    
    # Output options
    parser.add_argument('--show-progress', action='store_true', help='Show progress bar')
    parser.add_argument('--generate-reports', action='store_true', help='Generate individual reports')
    parser.add_argument('--share-data-provider', action='store_true', help='Share data provider across workers')
    
    return parser.parse_args()

def print_system_info():
    """Print system information for debugging."""
    logger.info(f"CPU Count: {multiprocessing.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Python Version: {'.'.join(map(str, tuple(sys.version_info)[:3]))}")

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    global args
    args = parse_args()
    
    # Print system information
    print_system_info()
    
    # Get symbols to backtest
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.symbol_file:
        with open(args.symbol_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
    elif args.all_symbols:
        data_dir = Path('data')
        symbols = get_market_symbols(data_dir, args.market)
    else:
        # Default symbols based on selected market
        if args.market == 'hkex':
            symbols = ['0700.HK', '9988.HK', '0001.HK', '0005.HK', '0011.HK']
        elif args.market == 'nasdaq':
            symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        elif args.market == 'nyse':
            symbols = ['JPM', 'BAC', 'WMT', 'PG', 'JNJ']
        elif args.market == 'asx':
            symbols = ['BHP.AX', 'CBA.AX', 'NAB.AX', 'CSL.AX', 'WBC.AX']
    
    # Auto-detect concurrent tasks
    if args.max_concurrent is None and args.execution_mode == 'async':
        args.max_concurrent = multiprocessing.cpu_count() * 2
        print(f"Auto-detected {args.max_concurrent} concurrent tasks")
    
    # Run market backtest
    run_market_backtest(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        execution_mode=args.execution_mode,
        workers=args.workers,
        max_concurrent=args.max_concurrent,
        show_progress=args.show_progress,
        generate_reports=args.generate_reports,
        share_data_provider=args.share_data_provider,
        strategies=args.strategies,
        fast_period=args.fast_period,
        slow_period=args.slow_period,
        rsi_period=args.rsi_period,
        oversold_threshold=args.oversold_threshold,
        overbought_threshold=args.overbought_threshold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        position_size=args.position_size,
        slippage_model=args.slippage_model,
        slippage_value=args.slippage_value,
        commission_model=args.commission_model,
        commission_value=args.commission_value,
        initial_capital=args.initial_capital,
        market=args.market
    )
    
    print(f"Market backtest completed for {len(symbols)} symbols")

if __name__ == "__main__":
    main() 
"""
Example script demonstrating how to use the backtesting system.
"""

import os
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.market_system.backtesting.backtest_engine import BacktestEngine
from src.market_system.backtesting.data_provider import DataProvider
from src.market_system.backtesting.portfolio_manager import PortfolioManager
from src.market_system.backtesting.execution_handler import ExecutionHandler
from src.market_system.backtesting.strategies import TrendFollowingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_backtest():
    """Run a sample backtest."""
    # Define paths
    data_dir = Path("data")  # Root data directory
    reports_dir = Path("reports/backtest")
    
    # Ensure directories exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_provider = DataProvider(str(data_dir))
    strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
    portfolio_manager = PortfolioManager(initial_capital=Decimal("100000"))
    execution_handler = ExecutionHandler(
        slippage_model="percentage",
        slippage_params={"percentage": 0.001},
        commission_model="fixed",
        commission_params={"fixed_amount": Decimal("5.0")}
    )
    
    # Define backtest parameters
    end_date = datetime(2023, 12, 31)  # Use a specific end date
    start_date = datetime(2023, 1, 1)   # Use a specific start date
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=Decimal("100000"),
        start_date=start_date,
        end_date=end_date,
        data_provider=data_provider,
        strategies=[strategy],
        portfolio_manager=portfolio_manager,
        execution_handler=execution_handler
    )
    
    # Run backtest
    results = engine.run(
        symbols=["AAPL"],
        timeframe="1d",
        risk_percentage=1.0,
        market="nasdaq",  # Specify the market
        show_progress=True
    )
    
    # Generate report
    report_paths = engine.generate_report(
        output_dir=str(reports_dir),
        report_name="trend_following_example"
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

def create_sample_data(file_path: Path):
    """Create sample OHLCV data for testing."""
    print(f"Creating sample data at {file_path}")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with a trend and some volatility
    n = len(dates)
    base_price = 150.0
    trend = 0.0002  # Slight upward trend
    
    # Generate random walk with drift
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(trend, 0.015, n)
    prices = base_price * (1 + returns).cumprod()
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'open': prices * (1 + np.random.normal(0, 0.003, n)),
        'high': prices * (1 + np.random.normal(0.005, 0.005, n)),
        'low': prices * (1 - np.random.normal(0.005, 0.005, n)),
        'close': prices,
        'volume': np.random.randint(5000000, 50000000, n)
    })
    
    # Ensure high >= open, close, low and low <= open, close
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Save to CSV
    data.to_csv(file_path, index=False)
    print(f"Sample data created with {len(data)} rows")

if __name__ == "__main__":
    run_backtest() 
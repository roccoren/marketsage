"""
StockScan Backtesting System

A comprehensive backtesting framework for trading strategies, designed to simulate
and evaluate trading performance using historical market data.
"""

from src.market_system.backtesting.backtest_engine import BacktestEngine
from src.market_system.backtesting.data_provider import DataProvider
from src.market_system.backtesting.execution_handler import ExecutionHandler
from src.market_system.backtesting.portfolio_manager import PortfolioManager
from src.market_system.backtesting.performance_metrics import PerformanceMetrics
from src.market_system.backtesting.visualization import Visualization
from src.market_system.backtesting.report_generator import ReportGenerator

__all__ = [
    'BacktestEngine',
    'DataProvider',
    'ExecutionHandler',
    'PortfolioManager',
    'PerformanceMetrics',
    'Visualization',
    'ReportGenerator'
]

__version__ = '0.1.0' 
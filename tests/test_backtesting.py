import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.market_system.backtesting.data_provider import DataProvider
from src.market_system.backtesting.execution_handler import ExecutionHandler
from src.market_system.backtesting.portfolio_manager import PortfolioManager
from src.market_system.backtesting.performance_metrics import PerformanceMetrics
from src.market_system.backtesting.backtest_engine import BacktestEngine


class TestBacktestingSystem(unittest.TestCase):
    """Test cases for the backtesting system."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create test data directory
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create sample data
        self.create_sample_data()
        
        # Initialize components
        self.data_provider = DataProvider(str(self.test_data_dir))
        self.execution_handler = ExecutionHandler()
        self.portfolio_manager = PortfolioManager(initial_capital=Decimal("10000.0"))
        
        # Define test strategy
        class TestStrategy:
            def __init__(self, name="TestStrategy"):
                self.name = name
            
            def generate_signals(self, market_data, market_state=None):
                """Simple strategy that buys when price increases by 2% and sells when it decreases by 2%."""
                signals = []
                
                if len(market_data) < 2:
                    return signals
                
                current_price = market_data.iloc[-1]['close']
                prev_price = market_data.iloc[-2]['close']
                
                if current_price > prev_price * 1.02:
                    signals.append({
                        'symbol': market_data.iloc[-1]['symbol'],
                        'timestamp': market_data.iloc[-1]['timestamp'],
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': 1,
                        'strategy': self.name
                    })
                elif current_price < prev_price * 0.98:
                    signals.append({
                        'symbol': market_data.iloc[-1]['symbol'],
                        'timestamp': market_data.iloc[-1]['timestamp'],
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': 1,
                        'strategy': self.name
                    })
                
                return signals
        
        self.test_strategy = TestStrategy()
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            initial_capital=Decimal("10000.0"),
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            data_provider=self.data_provider,
            strategies=[self.test_strategy],
            portfolio_manager=self.portfolio_manager,
            execution_handler=self.execution_handler
        )

    def tearDown(self):
        """Clean up after each test."""
        # Remove test data files
        for file in self.test_data_dir.glob("*.csv"):
            file.unlink()
        
        # Remove test data directory
        self.test_data_dir.rmdir()

    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        # Generate 30 days of sample data
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=i) for i in range(31)]
        
        # Create sample data with some price movements
        data = []
        base_price = 100.0
        
        for i, date in enumerate(dates):
            # Add some randomness and trend to price
            price_change = np.random.normal(0, 1) + 0.1  # Slight upward bias
            base_price *= (1 + price_change / 100)
            
            # Create OHLCV data
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = base_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = base_price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.normal(1000000, 200000))
            
            # Ensure high is highest and low is lowest
            high_price = max(open_price, high_price, close_price)
            low_price = min(open_price, low_price, close_price)
            
            data.append({
                'timestamp': date,
                'symbol': 'TEST',
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(self.test_data_dir / "TEST.csv", index=False)

    def test_backtest_execution(self):
        """Test that backtest runs without errors."""
        # Run backtest
        results = self.backtest_engine.run(
            symbols=["TEST"],
            timeframe="1d",
            risk_percentage=1.0
        )
        
        # Check that results contain expected components
        self.assertIsNotNone(results)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Check that equity curve is not empty
        self.assertGreater(len(results['equity_curve']), 0)
        
        # Check that performance metrics were calculated
        self.assertIn('total_return', results['metrics'])
        self.assertIn('sharpe_ratio', results['metrics'])
        self.assertIn('max_drawdown', results['metrics'])

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample equity curve
        dates = [datetime.now() + timedelta(days=i) for i in range(10)]
        equity_values = [10000, 10100, 10200, 10150, 10300, 10250, 10400, 10500, 10450, 10600]
        
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'equity': equity_values
        })
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(equity_curve)
        
        # Check metrics
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # Verify total return calculation
        expected_return = (equity_values[-1] / equity_values[0]) - 1
        self.assertAlmostEqual(metrics['total_return'], expected_return, places=4)


if __name__ == '__main__':
    unittest.main() 
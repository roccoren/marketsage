import sys
import logging
from pathlib import Path
# Add src directory to path
src_path = str(Path(__file__).parent.parent.absolute())
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"Added {src_path} to Python path")

from decimal import Decimal
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
print(f"Relative import debug - __name__: {__name__}")
print(f"Relative import debug - __package__: {__package__}")
from market_system.trading_system import TradingSystem
from market_system.strategy.trend_following_strategy import TrendFollowingStrategy

def generate_sample_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)  # For reproducibility
    
    dates = [
        datetime.now() - timedelta(days=i)
        for i in range(days, 0, -1)
    ]
    
    # Generate price data with a trend
    base_price = 100
    trend = np.cumsum(np.random.normal(0.001, 0.02, days))
    noise = np.random.normal(0, 0.01, days)
    prices = base_price * (1 + trend + noise)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.002, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, days))),
        'close': prices,
        'volume': np.random.normal(1000000, 200000, days).astype(int)
    }, index=dates)
    
    # Ensure high/low prices are consistent
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Force immediate flushing of stdout
    sys.stdout.flush()
    
    # Initialize trading system with 100,000 initial capital
    trading_system = TradingSystem(Decimal('100000'))
    
    # Add trend following strategy
    strategy = TrendFollowingStrategy()
    trading_system.add_strategy(strategy)
    
    # Generate sample data for a few symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    market_data = {}
    
    for symbol in symbols:
        print(f"\nProcessing data for {symbol}...")
        sys.stdout.flush()
        
        # Generate sample data
        df = generate_sample_data(symbol)
        
        # Convert to MarketData objects
        from market_system.domain.models import MarketData
        
        market_data[symbol] = [
            MarketData(
                symbol=symbol,
                timestamp=index,
                open_price=Decimal(str(row['open'])),
                high_price=Decimal(str(row['high'])),
                low_price=Decimal(str(row['low'])),
                close_price=Decimal(str(row['close'])),
                volume=int(row['volume']),
                indicators={}
            )
            for index, row in df.iterrows()
        ]
        
        # Process market data
        trading_system.process_market_data(symbol, market_data[symbol])
        
        # Print current market state
        state = trading_system.get_market_state(symbol)
        print(f"Current market state for {symbol}: {state.name if state else 'Unknown'}")
        sys.stdout.flush()
    
    # Print current positions
    positions = trading_system.get_current_positions()
    if positions:
        print("\nCurrent Positions:")
        for symbol, position in positions.items():
            print(f"{symbol}: {position.quantity} @ {position.entry_price}")
    else:
        print("\nNo open positions")
    sys.stdout.flush()
    
    # Print risk metrics
    risk_metrics = trading_system.get_risk_metrics()
    print("\nRisk Metrics:")
    print(f"Total Position Size: {risk_metrics.position_size}")
    print(f"Current Exposure: {risk_metrics.current_exposure:.2%}")
    print(f"Value at Risk (95%): {risk_metrics.var_95}")
    print(f"Current Volatility: {risk_metrics.current_volatility:.2%}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
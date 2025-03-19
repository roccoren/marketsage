#!/usr/bin/env python3

import json
import sys
import os
from datetime import datetime

# Add the repo root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.market_system.market_backtest import initialize_strategies, run_market_backtest

def main():
    # Load a subset of symbols from the original file
    with open('config/market/mid-mrk-momentum-master.json', 'r') as f:
        data = json.load(f)
        symbols = data['symbols'][:10]  # Just use the first 10 symbols
    
    print(f"Running backtest with {len(symbols)} symbols")
    
    # Initialize strategies with more conservative parameters
    strategies = initialize_strategies(
        ['trend_following'],
        fast_period=10,
        slow_period=30,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        volume_threshold=1.5,
        volatility_threshold=1.5,
        consolidation_threshold=0.3
    )
    
    # Run backtest with more conservative risk parameters
    results = run_market_backtest(
        symbols=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 10),
        timeframe='1d',
        execution_mode='parallel',
        workers=2,
        show_progress=True,
        generate_reports=True,
        strategies=strategies,
        # More conservative parameters
        fast_period=10,
        slow_period=30,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        # Reduced risk parameters
        stop_loss=0.02,  # 2% stop loss
        take_profit=0.05,  # 5% take profit
        position_size=0.01,  # 1% position size
        use_influxdb=True,
        symbol_file='config/market/mid-mrk-momentum-master.json',
        generate_symbol_reports=True
    )
    
    print("Backtest completed successfully!")
    
if __name__ == "__main__":
    main() 
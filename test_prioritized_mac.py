#!/usr/bin/env python3

import json
import sys
import os
from datetime import datetime
from decimal import Decimal

# Add the repo root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.market_system.market_backtest import initialize_strategies, run_market_backtest

def main():
    # Load symbols from the original file
    with open('config/market/mid-mrk-momentum-master.json', 'r') as f:
        data = json.load(f)
        symbols = data['symbols'][:20]  # Use the first 20 symbols
    
    print(f"Running prioritized MAC backtest with {len(symbols)} symbols")
    
    # Set strategies in order of priority
    strategy_list = [
        'moving_average_crossover',  # Primary strategy
        'trend_following',
        'volatility_breakout',
        'mean_reversion',
        'volume_spike'
    ]
    
    # Initialize strategies with the correct priority order
    strategies = initialize_strategies(
        strategy_list,
        # Moving average parameters
        fast_period=10,
        slow_period=30,
        # RSI parameters
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        # Volume and volatility thresholds
        volume_threshold=1.5,
        volatility_threshold=1.5,
        consolidation_threshold=0.3
    )
    
    print(f"Initialized {len(strategies)} strategies with MAC as priority")
    
    # Run backtest with MAC prioritization
    results = run_market_backtest(
        symbols=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 10),
        timeframe='1d',
        execution_mode='parallel',
        workers=4,
        show_progress=True,
        generate_reports=True,
        strategies=strategy_list,  # Pass strategy names not instances
        # Conservative parameters
        fast_period=10,
        slow_period=30,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        # Safer risk parameters
        stop_loss=0.02,  # 2% stop loss
        take_profit=0.05,  # 5% take profit
        position_size=0.01,  # 1% position size
        use_influxdb=True,
        symbol_file='config/market/mid-mrk-momentum-master.json',
        generate_symbol_reports=True
    )
    
    print("Prioritized MAC backtest completed successfully!")
    
if __name__ == "__main__":
    main() 
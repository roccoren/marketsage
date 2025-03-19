#!/usr/bin/env python3

import json
import sys
import os
from datetime import datetime
from decimal import Decimal

# Add the repo root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.market_system.market_backtest import run_market_backtest

def main():
    # Load symbols from the original file
    with open('config/market/mid-mrk-momentum-master.json', 'r') as f:
        data = json.load(f)
        symbols = data['symbols'][:20]  # Use the first 20 symbols
    
    print(f"Running enhanced MAC backtest with {len(symbols)} symbols")
    
    # Set strategies with Moving Average Crossover as primary
    strategy_list = [
        'moving_average_crossover',  # Primary strategy
        'trend_following',
        'volatility_breakout',
        'mean_reversion',
        'volume_spike'
    ]
    
    # Run backtest with enhanced MAC parameters
    results = run_market_backtest(
        symbols=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 10),
        timeframe='1d',
        execution_mode='parallel',
        workers=4,
        show_progress=True,
        generate_reports=True,
        strategies=strategy_list,
        # Basic technical indicator parameters
        fast_period=10,
        slow_period=30,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        # Volatility-based risk management
        stop_loss=0.02,
        take_profit=0.05,
        position_size=0.01,
        # Additional MAC-specific parameters
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_period=14,
        adaptive_params=True,
        use_market_regime=True,
        # Other parameters
        use_influxdb=True,
        symbol_file='config/market/mid-mrk-momentum-master.json',
        generate_symbol_reports=True
    )
    
    print("Enhanced MAC backtest completed successfully!")
    
    # Calculate summary statistics
    total_return = 0
    profitable_symbols = 0
    total_symbols = len(results)
    
    if isinstance(results, list):
        for result in results:
            symbol_return = result.get('metrics', {}).get('total_return', 0) * 100
            print(f"Symbol {result.get('symbol')}: Return {symbol_return:.2f}%")
            total_return += symbol_return
            if symbol_return > 0:
                profitable_symbols += 1
    
        avg_return = total_return / total_symbols if total_symbols > 0 else 0
        win_rate = profitable_symbols / total_symbols * 100 if total_symbols > 0 else 0
        
        print(f"\nSummary Statistics:")
        print(f"Average Return: {avg_return:.2f}%")
        print(f"Profitable Symbols: {profitable_symbols}/{total_symbols} ({win_rate:.2f}%)")
    
if __name__ == "__main__":
    main() 
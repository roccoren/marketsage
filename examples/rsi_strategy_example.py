"""Example usage of RSI strategy."""

import pandas as pd
import yfinance as yf
from src.market_system.strategies.rsi_strategy import RSIStrategy

def main():
    # Download historical data
    symbol = "AAPL"
    data = yf.download(symbol, start="2023-01-01", end="2024-01-01")
    
    # Initialize strategy
    strategy = RSIStrategy(
        period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        stop_loss=0.02,
        take_profit=0.05,
        position_size=0.1
    )
    
    # Calculate signals
    signals = strategy.calculate_signals(data)
    
    # Get latest trading decision
    decision = strategy.generate_signals(data)
    
    # Print results
    print(f"\nAnalysis for {symbol}:")
    print(f"Latest RSI: {decision['metadata']['rsi']:.2f}")
    print(f"Recommended action: {decision['action'].upper()}")
    print(f"Stop loss: {decision['stop_loss']*100:.1f}%")
    print(f"Take profit: {decision['take_profit']*100:.1f}%")
    print(f"Position size: {decision['position_size']*100:.1f}%")
    
    # Print last few signals
    print("\nRecent signals:")
    print(signals[['Close', 'rsi', 'signal', 'position']].tail())

if __name__ == "__main__":
    main() 
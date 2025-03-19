"""Example usage of RSI strategy with multiple symbols."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def calculate_rsi(data, periods=14):
    """Calculate RSI for a given price series."""
    delta = data.diff()
    
    # Make two series: one for lower closes and one for higher closes
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.ewm(com=periods-1, adjust=False).mean()
    ma_down = down.ewm(com=periods-1, adjust=False).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    
    return rsi

def analyze_stock(symbol, start_date, end_date):
    """Analyze a single stock using RSI."""
    try:
        # Download historical data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty or len(data) < 15:  # Need at least 15 days for RSI calculation
            print(f"Warning: Not enough data for {symbol}")
            return None
        
        # Calculate RSI
        rsi = calculate_rsi(data['Close'])
        
        # Get latest values
        latest_close = data['Close'].iloc[-1].item()
        latest_rsi = rsi.iloc[-1].item()
        daily_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100).item()
        
        # Determine signal
        if np.isnan(latest_rsi):
            signal = "N/A"
        elif latest_rsi < 30:
            signal = "BUY"
        elif latest_rsi > 70:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            'symbol': symbol,
            'close': latest_close,
            'rsi': latest_rsi if not np.isnan(latest_rsi) else 0.0,
            'signal': signal,
            'daily_change': daily_change
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def main():
    # List of symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Last 60 days for better RSI calculation
    
    print(f"\nAnalyzing stocks from {start_date.date()} to {end_date.date()}\n")
    print("Symbol  |  Price   |   RSI   | Daily Change |  Signal")
    print("-" * 55)
    
    for symbol in symbols:
        result = analyze_stock(symbol, start_date, end_date)
        if result:
            print(f"{result['symbol']:6} | ${result['close']:7.2f} | {result['rsi']:6.2f} | {result['daily_change']:+7.2f}% | {result['signal']:6}")

if __name__ == "__main__":
    main() 
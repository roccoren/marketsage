#!/usr/bin/env python3
"""
Script to test that we can load the downloaded data properly.
"""

from pathlib import Path
import pandas as pd
import numpy as np

class MarketDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.freq_dir = self.data_dir / "1d"
        
    def list_instruments(self):
        """Get list of all instruments."""
        with open(self.freq_dir / "instruments" / "all.txt", "r") as f:
            # Skip header
            next(f)
            return [line.strip().split("\t")[0] for line in f]
    
    def get_calendar(self):
        """Get list of all dates."""
        with open(self.freq_dir / "calendars" / "day.txt", "r") as f:
            return [line.strip() for line in f]
    
    def get_feature(self, symbol, feature, start_time=None, end_time=None):
        """Get feature data for a symbol."""
        df = pd.read_csv(self.freq_dir / "features" / feature / f"{symbol}.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        if start_time:
            df = df[df['date'] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df['date'] <= pd.to_datetime(end_time)]
            
        return df.set_index('date')

def main():
    # Initialize data loader
    data_dir = Path("data/market")
    print(f"Loading data from {data_dir.absolute()}")
    
    loader = MarketDataLoader(data_dir)
    
    # Get instrument list
    instruments = loader.list_instruments()
    print(f"Found {len(instruments)} instruments: {instruments[:5]}...")
    
    # Get stock data
    start_time = "2020-01-01"
    end_time = "2020-12-31"
    features = ["close", "open", "high", "low", "volume"]
    
    # Get data for a sample stock (PLTR)
    print(f"\nFetching data for PLTR from {start_time} to {end_time}:")
    data = {feature: loader.get_feature("PLTR", feature, start_time, end_time) 
            for feature in features}
    df = pd.DataFrame({f"${feature}": data[feature][feature] for feature in features})
    print(df.head())
    
    # Calculate market return
    print("\nCalculating returns for some sample stocks:")
    sample_stocks = ['PLTR', 'SOUN', 'MSTR', 'RKLB', 'PTON']
    returns = {}
    
    for stock in sample_stocks:
        df = loader.get_feature(stock, "close", start_time, end_time)
        returns[stock] = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # Compare returns
    print("\nComparing returns of sample stocks:")
    for stock, ret in returns.items():
        print(f"{stock}: {ret:.2f}%")
    
    print("\nData test completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Script to download and process stock data for Qlib
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def download_stock_data(symbols, start_date, end_date):
    """Download stock data using yfinance"""
    data = {}
    for symbol in symbols:
        try:
            print(f"Downloading data for {symbol}...")
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                data[symbol] = df
                print(f"Successfully downloaded data for {symbol}")
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    return data

def process_data_for_qlib(data, output_dir):
    """Process data into Qlib format"""
    # Create output directories
    features_dir = output_dir
    os.makedirs(features_dir, exist_ok=True)
    
    # Process each stock's data
    for symbol, df in data.items():
        try:
            # Reset index to get date as a column
            df = df.reset_index()
            
            # Rename columns to match Qlib format
            result_df = pd.DataFrame()
            result_df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            result_df['open'] = df['Open']
            result_df['high'] = df['High']
            result_df['low'] = df['Low']
            result_df['close'] = df['Close']
            result_df['volume'] = df['Volume']
            result_df['price_adj'] = df['Close'] # Using close as adjusted price for simplicity
            
            # Save to bin file
            output_file = os.path.join(features_dir, f"{symbol}.day.bin")
            result_df.to_pickle(output_file)
            
            print(f"Saved processed data for {symbol} to {output_file}")
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")

def main():
    # Read symbols from instruments file
    try:
        with open('data/instruments/all.txt', 'r') as f:
            # File is tab separated with no header
            symbols = [line.split('\t')[0].strip() for line in f if line.strip()]
        print(f"Found {len(symbols)} symbols to process")
    except Exception as e:
        print(f"Error reading symbols file: {e}")
        return
    
    # Set date range
    end_date = datetime.now()
    start_date = datetime(2010, 1, 1)  # Start from 2010
    
    # Download data
    print("Starting data download...")
    data = download_stock_data(symbols, start_date, end_date)
    
    if not data:
        print("No data was downloaded. Exiting.")
        return
    
    # Process data for Qlib
    print("Processing downloaded data...")
    process_data_for_qlib(data, 'data/features/day')
    
    print("Data download and processing completed!")

if __name__ == "__main__":
    main() 
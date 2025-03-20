#!/usr/bin/env python
"""
Script to download US stock data for Qlib
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
import pkg_resources
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('download_us_data')

# Check required packages
required_packages = ["qlib", "pandas", "yfinance", "tqdm"]
for package in required_packages:
    try:
        pkg_resources.require(package)
        logger.info(f"Package {package} is already installed")
    except pkg_resources.DistributionNotFound:
        logger.error(f"Required package {package} is not installed")
        sys.exit(1)

def read_symbols_file(file_path):
    """Read stock symbols from a file"""
    logger.info(f"Reading symbols from {file_path}")
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbols file: {e}")
        sys.exit(1)

def download_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return None

def prepare_us_data(target_dir='~/.qlib/qlib_data/us_data', symbols_file=None):
    """Download US stock data and prepare it for Qlib"""
    if not symbols_file:
        logger.error("Symbols file is required")
        sys.exit(1)
        
    target_dir = os.path.expanduser(target_dir)
    
    # Create the proper directory structure for Qlib
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'calendars', 'day'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'instruments'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'features'), exist_ok=True)
    
    logger.info(f"Creating US stock data at {target_dir}")
    logger.info(f"Clearing existing data in {target_dir}")
    
    # Read symbols from file
    symbols = read_symbols_file(symbols_file)
    if not symbols:
        logger.error("No symbols found in the input file")
        sys.exit(1)
    
    # Download data
    logger.info(f"Downloading data for {len(symbols)} tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    
    all_data = {}
    trading_dates = set()
    
    for symbol in tqdm(symbols):
        df = download_stock_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            all_data[symbol] = df
            trading_dates.update(df.index.date)
    
    if not all_data:
        logger.error("No data was downloaded for any symbols")
        sys.exit(1)
    
    # Create calendar.pkl
    logger.info("Creating calendar.pkl file...")
    trading_dates = sorted(list(trading_dates))
    with open(os.path.join(target_dir, 'calendar.pkl'), 'wb') as f:
        pickle.dump(trading_dates, f)
    
    # Create day frequency calendar
    with open(os.path.join(target_dir, 'calendars', 'day', 'calendar.pkl'), 'wb') as f:
        pickle.dump(trading_dates, f)
    
    # Create symbolic links
    os.symlink(
        os.path.join('calendars', 'day', 'calendar.pkl'),
        os.path.join(target_dir, 'day.pkl')
    )
    
    # Create instruments.pkl
    logger.info("Creating instruments.pkl file...")
    instruments = {symbol: {'type': 'CS'} for symbol in all_data.keys()}
    with open(os.path.join(target_dir, 'instruments.pkl'), 'wb') as f:
        pickle.dump(instruments, f)
    
    # Create feature files
    logger.info("Creating feature files...")
    for symbol in tqdm(all_data.keys()):
        symbol_dir = os.path.join(target_dir, 'features', symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        df = all_data[symbol]
        features = {
            'open': df['Open'].values,
            'close': df['Close'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'volume': df['Volume'].values,
        }
        
        for feature_name, feature_data in features.items():
            feature_file = os.path.join(symbol_dir, f'{feature_name}.day.bin')
            feature_data.astype(np.float32).tofile(feature_file)
    
    # Create frequency metadata
    logger.info("Creating frequency metadata...")
    freq_meta = {'day': 'day'}
    with open(os.path.join(target_dir, 'freq_meta.pkl'), 'wb') as f:
        pickle.dump(freq_meta, f)
    
    logger.info("Data preparation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Download US stock data for Qlib")
    parser.add_argument("--target_dir", default="~/.qlib/qlib_data/us_data", help="Target directory for data")
    parser.add_argument("--symbols_file", required=True, help="File containing stock symbols")
    
    args = parser.parse_args()
    prepare_us_data(args.target_dir, args.symbols_file)

if __name__ == "__main__":
    main() 
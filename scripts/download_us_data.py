#!/usr/bin/env python
"""
Script to download US stock data for Qlib
"""

import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path
import traceback
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("download_us_data")

# Check and install required packages
required_packages = ["qlib", "pandas", "yfinance", "tqdm", "lxml"]

def install_package(package):
    """Install a package if it's not available"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except Exception as e:
        logger.error(f"Failed to install {package}: {e}")
        return False

# Check and install required packages
for package in required_packages:
    try:
        __import__(package)
        logger.info(f"Package {package} is already installed")
    except ImportError:
        logger.info(f"Package {package} is not installed. Installing...")
        if not install_package(package):
            logger.error(f"Cannot proceed without {package}. Exiting.")
            sys.exit(1)

# Now import packages after ensuring they're installed
import qlib
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
try:
    from qlib.data import D
    from qlib.constant import REG_US
except ImportError:
    # Fallback for older versions of qlib
    REG_US = "us"

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
        return []

def download_stock_data(symbol, start_date, end_date):
    """Download stock data using yfinance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        # Check if we have a MultiIndex (newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            # Get the first level of column names (Price)
            data.columns = data.columns.get_level_values(0)
        
        # Create a mapping of possible column names to standardized names
        column_map = {
            'Open': 'open', 'OPEN': 'open', 'open': 'open',
            'High': 'high', 'HIGH': 'high', 'high': 'high',
            'Low': 'low', 'LOW': 'low', 'low': 'low',
            'Close': 'close', 'CLOSE': 'close', 'close': 'close',
            'Volume': 'volume', 'VOLUME': 'volume', 'volume': 'volume',
            'Adj Close': 'adjclose', 'ADJ CLOSE': 'adjclose', 'Adj_Close': 'adjclose', 'adj_close': 'adjclose', 'adjclose': 'adjclose'
        }
        
        # Rename columns to align with qlib format, but only for columns that exist
        new_columns = {}
        for old_col, new_col in column_map.items():
            if old_col in data.columns:
                new_columns[old_col] = new_col
        
        data = data.rename(columns=new_columns)
        
        # Make sure all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
            if len(missing_columns) == len(required_columns):
                # If all required columns are missing, something is wrong
                return None
        
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return None

def prepare_us_data(target_dir='~/.qlib/qlib_data/us_data', min_stocks=50, symbols_file=None):
    """Download US stock data and prepare it for Qlib"""
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Creating US stock data at {target_dir}")
    
    # Clear existing data if present
    if os.path.exists(target_dir):
        logger.info(f"Clearing existing data in {target_dir}")
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # Create the proper directory structure for Qlib
    os.makedirs(os.path.join(target_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "calendars"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "instruments"), exist_ok=True)
    
    # Common stock indices/tickers
    indices = ["SPY", "QQQ", "DIA"]  # S&P 500, NASDAQ, Dow Jones
    
    # Major tech stocks
    tech_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
    
    # Get S&P 500 stocks
    try:
        logger.info("Fetching S&P 500 components...")
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers = sp500['Symbol'].str.replace('.', '-').tolist()
        logger.info(f"Found {len(sp500_tickers)} S&P 500 tickers")
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        sp500_tickers = []
    
    # If we couldn't get S&P 500 companies, fallback to a list of major stocks
    if not sp500_tickers:
        major_stocks = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "BRK-B", 
            "JPM", "JNJ", "WMT", "PG", "XOM", "BAC", "CSCO", "PFE", "T", "KO", "DIS",
            "NFLX", "ADBE", "CRM", "AMD", "INTC", "IBM", "ORCL", "GS", "V", "MA",
            "HD", "MCD", "CMCSA", "VZ", "CVX", "PEP", "ABT", "MRK", "PM", "TMO",
            "COST", "UNH", "MDT", "HON", "QCOM", "LLY", "BA", "CAT", "MMM", "GE"
        ]
        all_tickers = list(set(indices + tech_stocks + major_stocks))
    else:
        # Use both S&P 500 and our specific lists
        all_tickers = list(set(indices + tech_stocks + sp500_tickers))
    
    # Read symbols from file if provided
    if symbols_file:
        file_symbols = read_symbols_file(symbols_file)
        if file_symbols:
            logger.info(f"Adding {len(file_symbols)} symbols from file")
            all_tickers = list(set(all_tickers + file_symbols))
    
    # Make sure we have at least minimum number of stocks (sample if too many)
    if len(all_tickers) > 200:  # Limit to manageable number if we have full S&P500
        logger.info(f"Sampling 200 stocks from {len(all_tickers)} total")
        # Always include indices and major tech
        must_include = indices + tech_stocks
        remaining = [t for t in all_tickers if t not in must_include]
        np.random.seed(42)  # For reproducibility
        selected = np.random.choice(remaining, 200 - len(must_include), replace=False).tolist()
        all_tickers = must_include + selected
    
    if len(all_tickers) < min_stocks:
        logger.warning(f"Only found {len(all_tickers)} tickers, which is less than the minimum {min_stocks}")
        
    # Setup dates - go back 5 years from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # ~5 years of data
    
    # Download data for all tickers
    logger.info(f"Downloading data for {len(all_tickers)} tickers...")
    all_data = {}
    trading_dates = set()
    
    for ticker in tqdm(all_tickers):
        data = download_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is not None and not data.empty:
            all_data[ticker] = data
            trading_dates.update(data.index.date)
    
    # Create instruments.pkl file
    logger.info("Creating instruments.pkl file...")
    instruments = sorted(list(all_data.keys()))
    instruments_df = pd.DataFrame(
        {
            "instrument": instruments,
            "start_time": start_date.strftime('%Y-%m-%d'),
            "end_time": end_date.strftime('%Y-%m-%d'),
        }
    )
    instruments_df.to_pickle(os.path.join(target_dir, "instruments.pkl"))
    
    # Create full list of instruments in the format qlib expects
    with open(os.path.join(target_dir, "instruments", "all.txt"), "w") as f:
        for ticker in instruments:
            f.write(f"{ticker}\n")
            
    # Also save it in CSI300 file (qlib often looks for this)
    with open(os.path.join(target_dir, "instruments", "csi300.txt"), "w") as f:
        for ticker in instruments:
            f.write(f"{ticker}\n")
    
    # Create calendar.pkl file
    logger.info("Creating calendar.pkl file...")
    calendar_dates = sorted(list(trading_dates))
    
    # Convert dates to pandas timestamps
    calendar_series = pd.Series([pd.Timestamp(d) for d in calendar_dates])
    
    # Save the main calendar
    calendar_series.to_pickle(os.path.join(target_dir, "calendar.pkl"))
    
    # Save with frequency designation for day frequency
    day_dir = os.path.join(target_dir, "calendars", "day")
    os.makedirs(day_dir, exist_ok=True)
    calendar_series.to_pickle(os.path.join(day_dir, "calendar.pkl"))
    
    # Also create 1d frequency which is often used by qlib
    oneday_dir = os.path.join(target_dir, "calendars", "1d")
    os.makedirs(oneday_dir, exist_ok=True)
    calendar_series.to_pickle(os.path.join(oneday_dir, "calendar.pkl"))
    
    # Create a symlink for backwards compatibility
    day_symlink = os.path.join(target_dir, "day.pkl")
    oneday_symlink = os.path.join(target_dir, "1d.pkl")
    
    if os.path.exists(day_symlink):
        os.remove(day_symlink)
    if os.path.exists(oneday_symlink):
        os.remove(oneday_symlink)
        
    os.symlink(os.path.join(day_dir, "calendar.pkl"), day_symlink)
    os.symlink(os.path.join(oneday_dir, "calendar.pkl"), oneday_symlink)
    
    # Create feature files for each ticker
    logger.info("Creating feature files...")
    for ticker, data in tqdm(all_data.items()):
        # Create ticker directory
        ticker_dir = os.path.join(target_dir, "features", ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Save each feature separately
        for col in data.columns:
            feature_data = data[col].copy()
            feature_data.to_pickle(os.path.join(ticker_dir, f"{col}.pkl"))
    
    # Create frequency metadata
    logger.info("Creating frequency metadata...")
    freq_meta = {
        "__DEFAULT_FREQ": "day",
        "day": {
            "freq": "day",
            "align": "right",
            "suffix": "day",
            "data_path": target_dir
        },
        "1d": {
            "freq": "1d",
            "align": "right",
            "suffix": "1d",
            "data_path": target_dir
        }
    }
    
    # Save frequency metadata
    freq_meta_path = os.path.join(target_dir, "freq_meta.pkl")
    with open(freq_meta_path, 'wb') as f:
        import pickle
        pickle.dump(freq_meta, f)
    
    logger.info("Data preparation completed successfully!")
    return target_dir

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download US stock data for Qlib")
    parser.add_argument("--target_dir", default="~/.qlib/qlib_data/us_data", help="Target directory for data")
    parser.add_argument("--min_stocks", type=int, default=50, help="Minimum number of stocks to download")
    parser.add_argument("--symbols_file", help="File containing additional stock symbols")
    args = parser.parse_args()
    
    prepare_us_data(args.target_dir, args.min_stocks, args.symbols_file)

if __name__ == "__main__":
    main() 
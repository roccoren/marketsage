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

def prepare_us_data(target_dir='~/.qlib/qlib_data/us_data', min_stocks=50):
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
    
    # Create instruments directory for qlib
    instruments_dir = os.path.join(target_dir, "instruments")
    os.makedirs(instruments_dir, exist_ok=True)
    
    # Create full list of instruments in the format qlib expects
    with open(os.path.join(instruments_dir, "all.txt"), "w") as f:
        for ticker in instruments:
            f.write(f"{ticker}\n")
            
    # Also save it in CSI300 file (qlib often looks for this)
    with open(os.path.join(instruments_dir, "csi300.txt"), "w") as f:
        for ticker in instruments:
            f.write(f"{ticker}\n")
    
    # Create calendar.pkl file
    logger.info("Creating calendar.pkl file...")
    calendar_dates = sorted(list(trading_dates))
    
    # Convert dates to pandas timestamps
    calendar_series = pd.Series([pd.Timestamp(d) for d in calendar_dates])
    
    # Save the main calendar
    calendar_series.to_pickle(os.path.join(target_dir, "calendar.pkl"))
    
    # Create proper directory structure for qlib frequencies
    # The key is to place calendars in the correct structure:
    # - frequency metadata directory
    os.makedirs(os.path.join(target_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "calendars"), exist_ok=True)
    
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
    
    # Remove existing files if they exist
    if os.path.exists(day_symlink):
        os.remove(day_symlink)
    if os.path.exists(oneday_symlink):
        os.remove(oneday_symlink)
    
    # Create symlinks
    try:
        os.symlink(os.path.join("calendars", "day", "calendar.pkl"), day_symlink)
        os.symlink(os.path.join("calendars", "1d", "calendar.pkl"), oneday_symlink)
    except Exception as e:
        # If symlinks fail, just copy the files
        logger.warning(f"Symlink creation failed: {e}")
        shutil.copy(
            os.path.join(target_dir, "calendars", "day", "calendar.pkl"), 
            day_symlink
        )
        shutil.copy(
            os.path.join(target_dir, "calendars", "1d", "calendar.pkl"), 
            oneday_symlink
        )
    
    # Create raw directory structure
    raw_dir = os.path.join(target_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create feature directory structure
    for ticker in tqdm(instruments, desc="Creating feature files"):
        ticker_path = os.path.join(raw_dir, ticker)
        os.makedirs(ticker_path, exist_ok=True)
        
        # Get data for this ticker
        data = all_data[ticker]
        
        # Create feature file with all required columns
        feature_df = pd.DataFrame(index=data.index)
        
        # Add basic OHLCV features with fallbacks for missing columns
        for col, qlib_col in [
            ('open', '$open'), 
            ('high', '$high'), 
            ('low', '$low'), 
            ('close', '$close'), 
            ('volume', '$volume')
        ]:
            if col in data.columns:
                feature_df[qlib_col] = data[col].values
            else:
                # Fallback: if a column is missing, try to derive it or use close
                if col == 'open' and 'close' in data.columns:
                    feature_df[qlib_col] = data['close'].values
                    logger.warning(f"{col} not available for {ticker}, using close price instead")
                elif col == 'high' and 'close' in data.columns:
                    feature_df[qlib_col] = data['close'].values
                    logger.warning(f"{col} not available for {ticker}, using close price instead")
                elif col == 'low' and 'close' in data.columns:
                    feature_df[qlib_col] = data['close'].values
                    logger.warning(f"{col} not available for {ticker}, using close price instead")
                elif col == 'volume':
                    feature_df[qlib_col] = 0  # Default volume to 0 if missing
                    logger.warning(f"{col} not available for {ticker}, using 0 instead")
                else:
                    # This should not happen if we properly filtered out stocks with no data
                    logger.error(f"Critical column {col} not available for {ticker} and no fallback available")
                    break
        
        # Add adjustment factor
        feature_df["$factor"] = 1.0  # Default adjustment factor
        
        # Handle case when adjclose isn't available
        if "adjclose" in data.columns:
            feature_df["$adjclose"] = data["adjclose"].values
        elif "close" in data.columns:
            # If adjclose is not available, use close price as fallback
            feature_df["$adjclose"] = data["close"].values
            logger.warning(f"Adjusted close not available for {ticker}, using close price instead")
        else:
            # This should not happen if we properly filtered out stocks with no data
            logger.error(f"Both adjclose and close are not available for {ticker}")
            continue
        
        # Calculate VWAP or fallback to close price
        if all(col in data.columns for col in ['high', 'low', 'close']):
            feature_df["$vwap"] = (data["high"] + data["low"] + data["close"]) / 3
        elif "close" in data.columns:
            feature_df["$vwap"] = data["close"]
            logger.warning(f"Cannot calculate VWAP for {ticker}, using close price instead")
        else:
            # This should not happen if we properly filtered out stocks with no data
            logger.error(f"Cannot calculate VWAP for {ticker} and no fallback available")
            continue
        
        # Save to pickle
        feature_df.to_pickle(os.path.join(ticker_path, "feature.pkl"))
    
    # Create a bin directory to store prepared data
    bin_dir = os.path.join(target_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    # Write a README file
    with open(os.path.join(target_dir, "README.md"), "w") as f:
        f.write("# US Stock Data for Qlib\n\n")
        f.write(f"This directory contains US stock data for use with Qlib.\n\n")
        f.write(f"- Data generated on: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- Number of stocks: {len(instruments)}\n")
        f.write(f"- Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        f.write(f"- Features: open, high, low, close, volume, adjclose, vwap\n\n")
    
    # Test the data with Qlib
    try:
        logger.info("Testing Qlib initialization...")
        qlib.init(provider_uri=target_dir, region=REG_US)
        
        # Test for a few stocks
        test_tickers = [t for t in ["AAPL", "MSFT", "NVDA"] if t in all_data]
        if test_tickers:
            try:
                from qlib.data import D
                logger.info(f"Testing data retrieval for {test_tickers[0]}...")
                # Use a slightly narrower date range for testing
                test_start = start_date + timedelta(days=10)
                test_end = end_date - timedelta(days=10)
                
                fields = ["$close", "$open", "$high", "$low", "$volume"]
                
                # Try with explicit frequency
                logger.info("Testing with explicit frequency (day)...")
                try:
                    df = D.features(
                        test_tickers[0:1], 
                        fields, 
                        test_start.strftime('%Y-%m-%d'),
                        test_end.strftime('%Y-%m-%d'),
                        freq="day"
                    )
                    
                    if df is not None and not df.empty:
                        logger.info(f"Successfully retrieved data with freq=day! Sample:\n{df.head()}")
                    else:
                        logger.warning("Retrieved empty dataframe with freq=day")
                except Exception as e:
                    logger.warning(f"Error testing with freq=day: {e}")
                    
                    # Try with 1d frequency
                    logger.info("Testing with explicit frequency (1d)...")
                    try:
                        df = D.features(
                            test_tickers[0:1], 
                            fields, 
                            test_start.strftime('%Y-%m-%d'),
                            test_end.strftime('%Y-%m-%d'),
                            freq="1d"
                        )
                        
                        if df is not None and not df.empty:
                            logger.info(f"Successfully retrieved data with freq=1d! Sample:\n{df.head()}")
                        else:
                            logger.warning("Retrieved empty dataframe with freq=1d")
                    except Exception as e:
                        logger.warning(f"Error testing with freq=1d: {e}")
                        
                        # If both explicit frequencies fail, try direct file access
                        logger.info("Testing direct file access...")
                        feature_path = os.path.join(target_dir, "raw", test_tickers[0], "feature.pkl")
                        if os.path.exists(feature_path):
                            df = pd.read_pickle(feature_path)
                            logger.info(f"Direct file access successful! Sample:\n{df.head()}")
                        else:
                            logger.warning(f"Feature file not found at {feature_path}")
                
            except Exception as e:
                logger.error(f"Error testing data retrieval: {e}")
                traceback.print_exc()
    except Exception as e:
        logger.error(f"Error initializing Qlib: {e}")
        traceback.print_exc()
    
    logger.info(f"US stock data successfully created at {target_dir}")
    logger.info(f"Downloaded data for {len(all_data)} stocks")
    
    return target_dir

if __name__ == "__main__":
    target_dir = '~/.qlib/qlib_data/us_data'
    
    # Allow custom target directory from command line
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    try:
        downloaded_dir = prepare_us_data(target_dir)
        if downloaded_dir:
            logger.info(f"Data preparation complete. You can now use the US stock data at: {downloaded_dir}")
            sys.exit(0)
        else:
            logger.error("Failed to prepare US stock data.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1) 
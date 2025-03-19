#!/usr/bin/env python3
"""
Script to download market data using yfinance.
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Custom Yahoo Finance data collector."""
    
    def __init__(self, save_dir, start_date, end_date=None, interval="1d", max_workers=4):
        self.save_dir = Path(save_dir)
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        self.max_workers = max_workers
        
        # Make sure base directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create frequency directory (1d for daily data)
        self.freq_dir = self.save_dir / "1d"
        self.freq_dir.mkdir(exist_ok=True)
        
        # Create subdirectories with correct structure
        (self.freq_dir / "raw").mkdir(exist_ok=True)
        (self.freq_dir / "calendars").mkdir(exist_ok=True)
        (self.freq_dir / "instruments").mkdir(exist_ok=True)
        (self.freq_dir / "features").mkdir(exist_ok=True)
    
    def download_symbol(self, symbol):
        """Download data for a single symbol."""
        try:
            logger.info(f"Downloading {symbol}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                actions=True  # Include dividends and stock splits
            )
            
            # Skip if empty
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Format data
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'split'
            })
            
            # Save raw data
            raw_dir = self.freq_dir / "raw"
            df.to_csv(raw_dir / f"{symbol}.csv", index=False)
            
            return symbol, df
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")
            return None
    
    def download_data(self, symbols):
        """Download data for multiple symbols using threads."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_symbol, symbol): symbol for symbol in symbols}
            
            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        symbol, df = result
                        results[symbol] = df
                    pbar.update(1)
        
        return results
    
    def normalize_data(self, symbols=None):
        """Normalize the downloaded data."""
        logger.info("Normalizing data...")
        
        # Create directories
        calendars_dir = self.freq_dir / "calendars"
        instruments_dir = self.freq_dir / "instruments"
        features_dir = self.freq_dir / "features"
        
        # Process each symbol
        raw_dir = self.freq_dir / "raw"
        all_dates = set()
        
        if symbols is None:
            # Get all symbols from raw directory
            symbols = [f.stem for f in raw_dir.glob("*.csv")]
        
        # Create instrument list
        instrument_rows = []
        for symbol in symbols:
            file_path = raw_dir / f"{symbol}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                if not df.empty:
                    start_date = df['date'].iloc[0].split(' ')[0]  # Extract date part only
                    end_date = df['date'].iloc[-1].split(' ')[0]   # Extract date part only
                    instrument_rows.append(f"{symbol}\t{start_date}\t{end_date}")
        
        # Write instrument list
        with open(instruments_dir / "all.txt", "w") as f:
            f.write("instrument\tstart_datetime\tend_datetime\n")  # Header
            for row in instrument_rows:
                f.write(f"{row}\n")
        
        # Process all dates
        for symbol in symbols:
            file_path = raw_dir / f"{symbol}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Extract date part only from datetime strings
                dates = [d.split(' ')[0] for d in df['date'].tolist()]
                all_dates.update(dates)
        
        # Write calendar
        all_dates = sorted(list(all_dates))
        with open(calendars_dir / "day.txt", "w") as f:
            for date in all_dates:
                f.write(f"{date}\n")
        
        # Process features
        features = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'split']
        
        for feature in features:
            feature_dir = features_dir / feature
            feature_dir.mkdir(exist_ok=True)
            
            for symbol in symbols:
                file_path = raw_dir / f"{symbol}.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        # Extract feature and date
                        if feature in df.columns:
                            # Create a proper copy to avoid SettingWithCopyWarning
                            feature_df = pd.DataFrame()
                            feature_df['date'] = df['date'].apply(lambda x: x.split(' ')[0])
                            feature_df[feature] = df[feature].values
                            feature_df.to_csv(feature_dir / f"{symbol}.csv", index=False)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} for {feature}: {str(e)}")
        
        logger.info("Data normalization completed!")

def load_symbols_from_file(file_path):
    """Load symbols from a text file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def download_market_data(symbols_file=None, symbols=None, save_dir=None, start_date=None, end_date=None, interval="1d"):
    """Download market data using custom collector."""
    # Define target directory for market data
    target_dir = Path(save_dir) if save_dir else Path("data/market")
    
    logger.info(f"Starting market data download to {target_dir.absolute()}...")
    
    # Get symbols either from direct input or from file
    symbol_list = []
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
        logger.info(f"Using {len(symbol_list)} symbols from command line: {symbols}")
    elif symbols_file:
        # Load symbols from file
        symbols_path = Path(symbols_file)
        symbol_list = load_symbols_from_file(symbols_path)
        logger.info(f"Loaded {len(symbol_list)} symbols from {symbols_path}")
    else:
        # Default symbols file
        symbols_path = Path("config/market/mid-mrk-momentum-master.txt")
        symbol_list = load_symbols_from_file(symbols_path)
        logger.info(f"Loaded {len(symbol_list)} symbols from default file: {symbols_path}")
    
    if not symbol_list:
        logger.error("No symbols specified. Please provide symbols directly or via a symbols file.")
        return None
    
    # Download data
    collector = MarketDataCollector(
        save_dir=target_dir,
        start_date=start_date or "2019-01-01",
        end_date=end_date,
        interval=interval,
        max_workers=8
    )
    
    # Download data
    collector.download_data(symbol_list)
    
    # Process data
    collector.normalize_data(symbol_list)
    
    logger.info("Market data download completed!")
    
    return target_dir

def main():
    try:
        # Download data
        data_dir = download_market_data()
        logger.info("Process completed!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Process completed!")

if __name__ == "__main__":
    main() 
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union, Any

# Import Qlib modules
import qlib
from qlib.data import D

logger = logging.getLogger(__name__)

class QlibSimpleInterface:
    """
    Simplified interface for using Qlib for data processing without ML dependencies
    """
    def __init__(self, region: str = "us"):
        """
        Initialize the Qlib interface.
        
        Args:
            region: Region code ('us', 'cn', etc.)
        """
        self.region = region
        self.initialized = False
        
    def init_qlib(self, data_path: Optional[str] = None):
        """
        Initialize Qlib with the specified configuration.
        
        Args:
            data_path: Path to the data directory
        """
        if self.initialized:
            return
            
        try:
            # Set up default data path if not provided
            if data_path is None:
                data_path = os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "us_data")
                os.makedirs(data_path, exist_ok=True)
            
            # Initialize Qlib
            if self.region.lower() == "cn":
                # For Chinese market data
                from qlib.config import REG_CN
                qlib.init(provider_uri=data_path, region=REG_CN)
            else:
                # For US market data
                qlib.init(provider_uri=data_path)
                
            self.initialized = True
            logger.info(f"Qlib initialized with data path: {data_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Qlib: {str(e)}")
            raise
            
    def download_data(self, symbols: List[str], start_date: str, end_date: Optional[str] = None,
                     target_dir: Optional[str] = None):
        """
        Download stock data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            target_dir: Target directory to save data
        """
        try:
            # Ensure Qlib is initialized
            if not self.initialized:
                self.init_qlib(target_dir)
                
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Create universe file
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
                universe_path = os.path.join(target_dir, "universe.txt")
                with open(universe_path, "w") as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
                logger.info(f"Created universe file with {len(symbols)} symbols at {universe_path}")
            
            # Download data using yahoofinance directly for simplicity
            import yfinance as yf
            
            # Download data for each symbol
            results = {}
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    
                    # Use yfinance directly for simplicity
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    # Reset index to make Date a column
                    df = df.reset_index()
                    
                    # Rename columns to standard format
                    df = df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'Close': 'close',
                        'High': 'high',
                        'Low': 'low',
                        'Volume': 'volume'
                    })
                    
                    # Convert date to string
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    
                    results[symbol] = df
                    logger.info(f"Downloaded {len(df)} records for {symbol}")
                    
                    # Save data if target directory is provided
                    if target_dir:
                        symbol_dir = os.path.join(target_dir, self.region.upper())
                        os.makedirs(symbol_dir, exist_ok=True)
                        
                        # Save as JSON for compatibility with existing code
                        records = df.to_dict('records')
                        
                        import json
                        with open(os.path.join(symbol_dir, f"{symbol}.json"), 'w') as f:
                            json.dump({'history': records}, f, indent=2)
                        
                        logger.info(f"Saved data to {os.path.join(symbol_dir, f'{symbol}.json')}")
                        
                except Exception as e:
                    logger.error(f"Error downloading data for {symbol}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in download_data: {str(e)}")
            raise 
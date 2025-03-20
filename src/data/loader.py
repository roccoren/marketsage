"""Data loader for stock data"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for stock data
    
    This class loads stock data from various sources and processes it for use
    in the trading system.
    """
    
    def __init__(self, cache_dir: str = "data/stocks"):
        """
        Initialize the data loader
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def load_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        include_technicals: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for the given symbols
        
        Parameters
        ----------
        symbols : List[str]
            List of stock symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        include_technicals : bool
            Whether to include technical indicators
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of DataFrames with stock data
        """
        result = {}
        
        for symbol in symbols:
            try:
                # Always download fresh data to avoid timezone issues with cached data
                df = self._download_data(symbol, start_date, end_date)
                
                # Process data
                if df is not None and not df.empty:
                    if include_technicals:
                        df = self._process_data(df)
                    result[symbol] = df
                    self.logger.info(f"Loaded data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
        
        return result
    
    def _download_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download stock data from yfinance
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from cache
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with stock data or None if not in cache
        """
        cache_file = self.cache_dir / f"{symbol}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Convert string dates to pandas Timestamp objects without timezone
                start_timestamp = pd.Timestamp(start_date).tz_localize(None)
                end_timestamp = pd.Timestamp(end_date).tz_localize(None)

                # Ensure index is timezone-naive as well
                # Check if index has timezone info (DatetimeIndex)
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Filter by date using timestamp objects
                df = df[(df.index >= start_timestamp) & (df.index <= end_timestamp)]
                if not df.empty:
                    return df
            except Exception as e:
                self.logger.error(f"Error loading {symbol} from cache: {e}")
        return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Save stock data to cache
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        df : pd.DataFrame
            DataFrame with stock data
        """
        if df is None or df.empty:
            return
        
        cache_file = self.cache_dir / f"{symbol}.csv"
        try:
            df.to_csv(cache_file)
        except Exception as e:
            self.logger.error(f"Error saving {symbol} to cache: {e}")
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process stock data, adding technical indicators
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with stock data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with processed data
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df["returns"] = df["Close"].pct_change()
        
        # RSI calculation
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        df["ma12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["ma26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ma12"] - df["ma26"]
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["hist"] = df["macd"] - df["signal"]
        
        # Moving averages
        df["ma10"] = df["Close"].rolling(window=10).mean()
        df["ma20"] = df["Close"].rolling(window=20).mean()
        df["ma50"] = df["Close"].rolling(window=50).mean()
        
        # Bollinger Bands
        df["bb_middle"] = df["ma20"]
        std = df["Close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std
        
        return df

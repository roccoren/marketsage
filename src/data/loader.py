from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
import logging
from ..config.settings import DATA_DIR

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, symbols, start_date=None, end_date=None, data_source="yfinance"):
        self.symbols = symbols
        self.start_date = start_date or datetime(2020, 1, 1)
        self.end_date = end_date or datetime.now()
        self.data_source = data_source
        self.data = {}

    def load_data(self):
        logger.info(f"Loading data for symbols: {self.symbols}")
        if self.data_source == "yfinance":
            self.data = self._load_yfinance_data()
        return self.data

    def _load_yfinance_data(self):
        data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                if not df.empty:
                    data[symbol] = self._process_data(df)
                    logger.info(f"Loaded data for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        return data

    def _process_data(self, df):
        df["returns"] = df["Close"].pct_change()
        df["ma10"] = df["Close"].rolling(10).mean()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma50"] = df["Close"].rolling(50).mean()
        
        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        
        return df

    def save_to_qlib_format(self):
        logger.info("Saving data in Qlib format")
        for symbol, df in self.data.items():
            symbol_dir = DATA_DIR / "qlib" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(symbol_dir / "day.csv")
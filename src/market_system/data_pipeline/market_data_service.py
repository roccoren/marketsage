from typing import Dict, List, Optional
from datetime import datetime
import logging
import pandas as pd
import pandas_ta as ta
from ..domain.models import MarketData

class MarketDataService:
    def __init__(self):
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_config = {
            'sma': [20, 50, 200],  # Simple Moving Averages
            'rsi': [14],           # Relative Strength Index
            'bbands': [20, 2],     # Bollinger Bands
            'macd': [12, 26, 9],   # MACD
            'atr': [14]            # Average True Range
        }

    def process_market_data(self, symbol: str, data: List[MarketData]) -> List[MarketData]:
        """
        Process raw market data and calculate technical indicators.
        
        Args:
            symbol: Trading symbol
            data: List of MarketData objects
        
        Returns:
            List of processed MarketData objects
        """
        logging.info(f"Processing market data for {symbol}")
        logging.info(f"Input data type: {type(data)}")
        
        # Convert List[MarketData] to DataFrame
        data = self._to_dataframe(data)
        # Ensure data is properly sorted
        data = data.sort_index()
        
        # Calculate technical indicators
        processed_data = self._calculate_indicators(data)
        
        # Update cache
        self._data_cache[symbol] = processed_data
        
        # Convert to domain objects
        return self._to_market_data(symbol, processed_data)

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on configuration."""
        df = data.copy()
        
        # Calculate SMAs
        for period in self._indicators_config['sma']:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        
        # Calculate RSI
        df[f'rsi_{self._indicators_config["rsi"][0]}'] = ta.rsi(
            df['close'], length=self._indicators_config['rsi'][0]
        )
        
        # Calculate Bollinger Bands
        bb_length, bb_std = self._indicators_config['bbands']
        bb = ta.bbands(df['close'], length=bb_length, std=bb_std)
        df = pd.concat([df, bb], axis=1)
        
        # Calculate MACD
        fast, slow, signal = self._indicators_config['macd']
        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        df = pd.concat([df, macd], axis=1)
        
        # Calculate ATR
        df[f'atr_{self._indicators_config["atr"][0]}'] = ta.atr(
            df['high'], df['low'], df['close'], length=self._indicators_config['atr'][0]
        )
        
        return df

    def _to_market_data(self, symbol: str, data: pd.DataFrame) -> List[MarketData]:
        """Convert processed DataFrame to list of MarketData objects."""
        market_data_list = []
        
        for index, row in data.iterrows():
            indicators = {
                col: row[col] 
                for col in data.columns 
                if col not in ['open', 'high', 'low', 'close', 'volume']
            }
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=index,
                open_price=row['open'],
                high_price=row['high'],
                low_price=row['low'],
                close_price=row['close'],
                volume=row['volume'],
                indicators={k: v for k, v in indicators.items() if pd.notna(v)}
            )
            market_data_list.append(market_data)
        
        return market_data_list

    def get_latest_data(self, symbol: str, lookback: int = 100) -> Optional[List[MarketData]]:
        """
        Retrieve latest market data for a symbol.
        
        Args:
            symbol: Trading symbol
            lookback: Number of historical data points to retrieve
            
        Returns:
            List of MarketData objects or None if data not available
        """
        if symbol not in self._data_cache:
            return None
            
        data = self._data_cache[symbol].tail(lookback)
        return self._to_market_data(symbol, data)

    def _to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert list of MarketData objects to DataFrame."""
        df_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        timestamps = []
        
        for item in data:
            timestamps.append(item.timestamp)
            df_data['open'].append(float(item.open_price))
            df_data['high'].append(float(item.high_price))
            df_data['low'].append(float(item.low_price))
            df_data['close'].append(float(item.close_price))
            df_data['volume'].append(item.volume)
        
        return pd.DataFrame(df_data, index=timestamps)
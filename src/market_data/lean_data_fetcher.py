"""
Stock data fetcher using QuantConnect API.
This module provides functionality to fetch stock data from various exchanges including
NASDAQ, NYSE, ASX, HKEX, TSX (Canada), and SGX (Singapore).
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import pandas as pd
import requests
from requests.exceptions import RequestException

class Resolution(Enum):
    """Data resolution options."""
    Daily = "daily"
    Hour = "hour"
    Minute = "minute"
    Second = "second"

class LeanDataFetcher:
    """Fetches stock data using QuantConnect's API."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            api_key (str, optional): QuantConnect API key
            api_secret (str, optional): QuantConnect API secret
        """
        self.api_key = api_key or os.getenv('QUANTCONNECT_API_KEY')
        self.api_secret = api_secret or os.getenv('QUANTCONNECT_API_SECRET')
        self.base_url = "https://www.quantconnect.com/api/v2"
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "QuantConnect API credentials not found. Please provide them either "
                "through constructor arguments or set QUANTCONNECT_API_KEY and "
                "QUANTCONNECT_API_SECRET environment variables."
            )
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """Make an API request to QuantConnect."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Authorization': f"Bearer {self.api_key}:{self.api_secret}",
            'Accept': 'application/json'
        }
        
        try:
            response = requests.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def fetch_stock_data(
        self,
        symbol: str,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
        resolution: Resolution = Resolution.Daily,
        include_dividends: bool = True,
        include_splits: bool = True,
        adjust_prices: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch stock data for a given symbol and exchange.
        
        Args:
            symbol (str): Stock symbol
            exchange (str): Exchange code
            start_date (datetime): Start date
            end_date (datetime): End date
            resolution (Resolution): Data resolution
            include_dividends (bool): Whether to include dividend data
            include_splits (bool): Whether to include split data
            adjust_prices (bool): Whether to return adjusted prices
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Price, dividend, and split data
        """
        params = {
            'symbol': symbol,
            'exchange': exchange,
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'resolution': resolution.value,
            'include_dividends': str(include_dividends).lower(),
            'include_splits': str(include_splits).lower(),
            'adjust_prices': str(adjust_prices).lower()
        }
        
        data = self._make_request('data/equity', params=params)
        
        # Process price data
        price_data = []
        if 'prices' in data:
            for p in data['prices']:
                price_data.append({
                    'DateTime': pd.to_datetime(p['time']),
                    'Open': float(p['open']),
                    'High': float(p['high']),
                    'Low': float(p['low']),
                    'Close': float(p['close']),
                    'Volume': int(p['volume'])
                })
        
        # Process dividend data
        dividend_data = []
        if 'dividends' in data:
            for d in data['dividends']:
                dividend_data.append({
                    'DateTime': pd.to_datetime(d['time']),
                    'Amount': float(d['amount']),
                    'Currency': d.get('currency', 'USD')
                })
        
        # Process split data
        split_data = []
        if 'splits' in data:
            for s in data['splits']:
                split_data.append({
                    'DateTime': pd.to_datetime(s['time']),
                    'SplitFactor': float(s['split_factor'])
                })
        
        # Convert to DataFrames
        price_df = pd.DataFrame(price_data)
        dividend_df = pd.DataFrame(dividend_data)
        split_df = pd.DataFrame(split_data)
        
        # Set index
        for df in [price_df, dividend_df, split_df]:
            if not df.empty:
                df.set_index('DateTime', inplace=True)
                df.sort_index(inplace=True)
        
        return price_df, dividend_df, split_df
    
    def get_available_symbols(self, exchange: str) -> List[Dict[str, str]]:
        """
        Get list of available symbols for a given exchange.
        
        Args:
            exchange (str): Exchange code
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing symbol information
        """
        params = {'exchange': exchange}
        data = self._make_request('data/equity/symbols', params=params)
        
        symbols = []
        for symbol in data.get('symbols', []):
            symbols.append({
                'symbol': symbol['symbol'],
                'name': symbol.get('name', ''),
                'security_type': symbol.get('security_type', 'Equity'),
                'exchange': exchange
            })
        
        return symbols
    
    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """
        Get list of supported exchanges.
        
        Returns:
            List[str]: List of supported exchange codes
        """
        return [
            'NYSE',     # New York Stock Exchange
            'NASDAQ',   # NASDAQ
            'ASX',      # Australian Securities Exchange
            'HKEX',     # Hong Kong Stock Exchange
            'TSX',      # Toronto Stock Exchange
            'SGX'       # Singapore Exchange
        ] 
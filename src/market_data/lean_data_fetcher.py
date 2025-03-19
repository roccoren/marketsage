"""
Stock data fetcher using QuantConnect API.
This module provides functionality to fetch stock data from various exchanges including
NASDAQ, NYSE, ASX, HKEX, TSX (Canada), and SGX (Singapore).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import pandas as pd
import requests
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Resolution(Enum):
    """Data resolution options."""
    Daily = "daily"
    Hour = "hour"
    Minute = "minute"
    Second = "second"

class LeanDataFetcherError(Exception):
    """Base exception for LeanDataFetcher errors."""
    pass

class APICredentialsError(LeanDataFetcherError):
    """Raised when API credentials are missing or invalid."""
    pass

class APIRequestError(LeanDataFetcherError):
    """Raised when an API request fails."""
    pass

class LeanDataFetcher:
    """Fetches stock data using QuantConnect's API."""
    
    def __init__(self, api_id: str = None, api_token: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            api_id (str, optional): QuantConnect API ID (User ID)
            api_token (str, optional): QuantConnect API Token
        
        Raises:
            APICredentialsError: If API credentials are missing or invalid
        """
        self.api_id = api_id or os.getenv('QUANTCONNECT_API_ID')
        self.api_token = api_token or os.getenv('QUANTCONNECT_API_TOKEN')
        self.base_url = "https://www.quantconnect.com/api/v2"
        
        if not self.api_id or not self.api_token:
            raise APICredentialsError(
                "QuantConnect API credentials not found. Please provide them either "
                "through constructor arguments or set environment variables:\n"
                "  export QUANTCONNECT_API_ID='your_api_id'\n"
                "  export QUANTCONNECT_API_TOKEN='your_api_token'\n\n"
                "You can get these from your QuantConnect account at:\n"
                "https://www.quantconnect.com/account#api-access"
            )
        
        # Verify credentials with a test API call
        try:
            self._verify_credentials()
        except APIRequestError as e:
            raise APICredentialsError(f"Invalid API credentials: {str(e)}")
    
    def _verify_credentials(self):
        """Verify API credentials with a test call."""
        try:
            response = self._make_request('authenticate')
            if not response.get('success'):
                raise APIRequestError("Authentication failed")
            logger.info("Successfully authenticated with QuantConnect API")
        except Exception as e:
            raise APIRequestError(f"Failed to verify credentials: {str(e)}")
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """
        Make an API request to QuantConnect.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            
        Returns:
            Dict: API response
            
        Raises:
            APIRequestError: If the request fails
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Authorization': f"{self.api_id}:{self.api_token}",
            'Accept': 'application/json'
        }
        
        try:
            logger.debug(f"Making {method} request to {endpoint}")
            response = requests.request(method, url, headers=headers, params=params)
            
            # Log response details for debugging
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            try:
                response_data = response.json()
            except ValueError:
                logger.error(f"Failed to parse JSON response: {response.text}")
                raise APIRequestError("Invalid JSON response from API")
            
            if not response.ok:
                error_msg = response_data.get('message', 'Unknown error')
                logger.error(f"API request failed: {error_msg}")
                raise APIRequestError(f"API request failed: {error_msg}")
            
            return response_data
            
        except RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise APIRequestError(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise APIRequestError(f"Unexpected error: {str(e)}")
    
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
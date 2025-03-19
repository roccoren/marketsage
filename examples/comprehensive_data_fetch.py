"""
Comprehensive example script demonstrating various scenarios for fetching stock data
using the LeanDataFetcher with QuantConnect API.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.market_data import LeanDataFetcher, Resolution
from src.market_data.lean_data_fetcher import APICredentialsError, APIRequestError

def setup_api_credentials():
    """
    Guide user through setting up API credentials.
    """
    print("\nQuantConnect API Setup Guide:")
    print("1. Go to https://www.quantconnect.com/account#api-access")
    print("2. Find your API ID (User ID) and generate an API Token if you haven't already")
    print("3. Set the following environment variables:\n")
    print("   export QUANTCONNECT_API_ID='your_api_id'")
    print("   export QUANTCONNECT_API_TOKEN='your_api_token'\n")
    print("Current settings:")
    print(f"  QUANTCONNECT_API_ID: {'*' * 8 if os.getenv('QUANTCONNECT_API_ID') else 'Not set'}")
    print(f"  QUANTCONNECT_API_TOKEN: {'*' * 8 if os.getenv('QUANTCONNECT_API_TOKEN') else 'Not set'}\n")

def fetch_multi_exchange_comparison(
    fetcher: LeanDataFetcher,
    symbols_dict: Dict[str, str],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, pd.DataFrame]:
    """
    Fetch and compare stocks from multiple exchanges.
    
    Args:
        fetcher: LeanDataFetcher instance
        symbols_dict: Dictionary mapping symbols to exchanges
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        Dictionary mapping symbols to their price DataFrames
    """
    results = {}
    
    for symbol, exchange in symbols_dict.items():
        try:
            price_df, div_df, split_df = fetcher.fetch_stock_data(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                adjust_prices=True
            )
            
            if not price_df.empty:
                # Normalize prices to compare relative performance
                first_close = price_df['Close'].iloc[0]
                price_df['Normalized'] = price_df['Close'] / first_close
                results[f"{symbol}_{exchange}"] = price_df
                
                print(f"\nSuccessfully fetched data for {symbol} from {exchange}")
                print(f"Data points: {len(price_df)}")
                if not div_df.empty:
                    print(f"Dividends: {len(div_df)}")
                if not split_df.empty:
                    print(f"Splits: {len(split_df)}")
                    
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange}: {str(e)}")
    
    return results

def analyze_market_correlation(price_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix between different markets.
    """
    # Create a DataFrame with daily returns for each market
    returns_df = pd.DataFrame()
    
    for symbol, df in price_dfs.items():
        returns_df[symbol] = df['Close'].pct_change()
    
    return returns_df.corr()

def plot_comparison(price_dfs: Dict[str, pd.DataFrame], title: str):
    """
    Plot normalized price comparison between different markets.
    """
    plt.figure(figsize=(12, 6))
    
    for symbol, df in price_dfs.items():
        plt.plot(df.index, df['Normalized'], label=symbol)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join('results', 'market_comparison.png'))
    plt.close()

def fetch_high_frequency_data(
    fetcher: LeanDataFetcher,
    symbol: str,
    exchange: str,
    start_date: datetime,
    end_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch high-frequency (minute) data for a symbol.
    """
    return fetcher.fetch_stock_data(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        resolution=Resolution.Minute,
        adjust_prices=True
    )

def main():
    """Main function demonstrating LEAN data fetching capabilities."""
    try:
        # Initialize the data fetcher
        logger.info("Initializing LeanDataFetcher...")
        fetcher = LeanDataFetcher()
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # One year of data
        
        print("\n1. Fetching symbols from different exchanges...")
        for exchange in fetcher.get_supported_exchanges():
            try:
                logger.info(f"Fetching symbols for {exchange}...")
                symbols = fetcher.get_available_symbols(exchange)
                print(f"\nSymbols from {exchange}:")
                print(f"Total symbols available: {len(symbols)}")
                if symbols:
                    print("Sample symbols:")
                    for symbol in symbols[:3]:
                        print(f"- {symbol['symbol']}: {symbol.get('name', 'N/A')}")
            except APIRequestError as e:
                logger.error(f"Failed to fetch symbols for {exchange}: {str(e)}")
        
        print("\n2. Comparing major market indices...")
        indices = {
            'SPY': 'NYSE',    # S&P 500 ETF
            'HSI': 'HKEX',    # Hang Seng Index
            'XJO': 'ASX',     # ASX 200
            'STI': 'SGX'      # Straits Times Index
        }
        
        for symbol, exchange in indices.items():
            try:
                logger.info(f"Fetching data for {symbol} from {exchange}...")
                price_df, div_df, split_df = fetcher.fetch_stock_data(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    adjust_prices=True
                )
                
                if not price_df.empty:
                    print(f"\n{symbol} data summary:")
                    print(f"Trading days: {len(price_df)}")
                    print(f"Price range: ${price_df['Low'].min():.2f} - ${price_df['High'].max():.2f}")
                    if not div_df.empty:
                        print(f"Dividends: {len(div_df)}")
                
            except APIRequestError as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
        
        print("\n3. Fetching high-frequency data for AAPL...")
        try:
            logger.info("Fetching AAPL minute data...")
            hf_start = end_date - timedelta(days=1)
            price_df, div_df, split_df = fetcher.fetch_stock_data(
                symbol='AAPL',
                exchange='NASDAQ',
                start_date=hf_start,
                end_date=end_date,
                resolution=Resolution.Minute
            )
            
            if not price_df.empty:
                print("\nHigh-frequency data summary:")
                print(f"Number of minutes: {len(price_df)}")
                print(f"Average volume per minute: {price_df['Volume'].mean():.0f}")
            
        except APIRequestError as e:
            logger.error(f"Failed to fetch AAPL minute data: {str(e)}")
        
        print("\n4. Analyzing dividend-paying stocks...")
        dividend_stocks = {
            'KO': 'NYSE',     # Coca-Cola
            'BHP': 'ASX',     # BHP Group
            'HSBC': 'HKEX'    # HSBC Holdings
        }
        
        for symbol, exchange in dividend_stocks.items():
            try:
                logger.info(f"Fetching dividend data for {symbol}...")
                price_df, div_df, split_df = fetcher.fetch_stock_data(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                    end_date=end_date,
                    include_dividends=True
                )
                
                if not div_df.empty:
                    print(f"\n{symbol} dividend summary:")
                    print(f"Number of dividends: {len(div_df)}")
                    print(f"Total dividend amount: ${div_df['Amount'].sum():.4f}")
                    print(f"Average dividend: ${div_df['Amount'].mean():.4f}")
                
            except APIRequestError as e:
                logger.error(f"Failed to fetch dividend data for {symbol}: {str(e)}")
    
    except APICredentialsError as e:
        logger.error(f"API credentials error: {str(e)}")
        setup_api_credentials()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
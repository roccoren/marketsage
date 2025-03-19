"""
Comprehensive example script demonstrating various scenarios for fetching stock data
using the LeanDataFetcher with QuantConnect API.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.market_data import LeanDataFetcher, Resolution

def check_api_credentials():
    """Check if API credentials are set in environment variables."""
    api_key = os.getenv('QUANTCONNECT_API_KEY')
    api_secret = os.getenv('QUANTCONNECT_API_SECRET')
    
    if not api_key or not api_secret:
        print("Error: QuantConnect API credentials not found!")
        print("Please set the following environment variables:")
        print("  QUANTCONNECT_API_KEY")
        print("  QUANTCONNECT_API_SECRET")
        print("\nYou can get these from your QuantConnect account settings.")
        return False
    return True

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
    if not check_api_credentials():
        return
        
    # Initialize the data fetcher
    fetcher = LeanDataFetcher()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    print("1. Fetching symbols from different exchanges...")
    for exchange in fetcher.get_supported_exchanges():
        print(f"\nSymbols from {exchange}:")
        symbols = fetcher.get_available_symbols(exchange)
        print(f"Total symbols available: {len(symbols)}")
        if symbols:
            print("Sample symbols:")
            for symbol in symbols[:3]:
                print(f"- {symbol['symbol']}: {symbol.get('name', 'N/A')}")
    
    print("\n2. Comparing major market indices...")
    indices = {
        'SPY': 'NYSE',    # S&P 500 ETF
        'HSI': 'HKEX',    # Hang Seng Index
        'XJO': 'ASX',     # ASX 200
        'STI': 'SGX'      # Straits Times Index
    }
    
    index_data = fetch_multi_exchange_comparison(
        fetcher,
        indices,
        start_date,
        end_date
    )
    
    if index_data:
        print("\nCalculating market correlations...")
        corr_matrix = analyze_market_correlation(index_data)
        print("\nMarket Correlations:")
        print(corr_matrix)
        
        print("\nPlotting market comparison...")
        plot_comparison(index_data, "Global Market Index Comparison")
    
    print("\n3. Fetching high-frequency data for AAPL...")
    try:
        # Fetch one day of minute data
        hf_start = end_date - timedelta(days=1)
        price_df, div_df, split_df = fetch_high_frequency_data(
            fetcher,
            'AAPL',
            'NASDAQ',
            hf_start,
            end_date
        )
        
        if not price_df.empty:
            print("\nHigh-frequency data summary:")
            print(f"Number of minutes: {len(price_df)}")
            print("\nFirst 5 minutes of trading:")
            print(price_df.head())
            
            # Calculate some intraday statistics
            print("\nIntraday Statistics:")
            print(f"Average minute volume: {price_df['Volume'].mean():.0f}")
            print(f"Highest minute volume: {price_df['Volume'].max():.0f}")
            print(f"Price range: ${price_df['High'].max() - price_df['Low'].min():.2f}")
    
    except Exception as e:
        print(f"Error fetching high-frequency data: {str(e)}")
    
    print("\n4. Analyzing dividend-paying stocks...")
    dividend_stocks = {
        'KO': 'NYSE',     # Coca-Cola
        'BHP': 'ASX',     # BHP Group
        'HSBC': 'HKEX'    # HSBC Holdings
    }
    
    for symbol, exchange in dividend_stocks.items():
        try:
            price_df, div_df, split_df = fetcher.fetch_stock_data(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                include_dividends=True,
                adjust_prices=True
            )
            
            if not div_df.empty:
                print(f"\n{symbol} Dividend Summary:")
                print(f"Number of dividends: {len(div_df)}")
                print(f"Total dividend amount: ${div_df['Amount'].sum():.4f}")
                print(f"Average dividend: ${div_df['Amount'].mean():.4f}")
                print("\nDividend Dates:")
                print(div_df.head())
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")

if __name__ == "__main__":
    main() 
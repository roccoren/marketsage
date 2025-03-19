"""
Example script demonstrating how to use the LeanDataFetcher to fetch stock data
from various exchanges, including adjusted prices and dividend data.
"""

from datetime import datetime, timedelta
import pandas as pd
from src.market_data.lean_data_fetcher import LeanDataFetcher

def print_available_symbols(fetcher: LeanDataFetcher, exchange: str):
    """Print available symbols for a given exchange."""
    print(f"\nAvailable symbols for {exchange}:")
    symbols = fetcher.get_available_symbols(exchange)
    for symbol_info in symbols[:5]:  # Print first 5 symbols
        name = symbol_info.get('name', 'N/A')
        print(f"- {symbol_info['symbol']}: {name}")
    if len(symbols) > 5:
        print(f"... and {len(symbols) - 5} more symbols")

def analyze_stock_data(price_df: pd.DataFrame, dividend_df: pd.DataFrame, split_df: pd.DataFrame, symbol: str):
    """Analyze and print stock data statistics."""
    print(f"\n{symbol} Stock Data Analysis:")
    
    if not price_df.empty:
        print("\nPrice Data (first 5 rows):")
        print(price_df.head())
        
        print("\nPrice Statistics:")
        print(f"Average Close Price: ${price_df['Close'].mean():.2f}")
        print(f"Highest Price: ${price_df['High'].max():.2f}")
        print(f"Lowest Price: ${price_df['Low'].min():.2f}")
        print(f"Total Trading Days: {len(price_df)}")
        
        # Calculate returns
        price_df['Daily_Return'] = price_df['Close'].pct_change()
        print(f"Average Daily Return: {price_df['Daily_Return'].mean():.4%}")
        print(f"Daily Return Volatility: {price_df['Daily_Return'].std():.4%}")
    
    if not dividend_df.empty:
        print("\nDividend History:")
        print(dividend_df.head())
        print(f"Total Dividends Paid: {len(dividend_df)}")
        if 'Amount' in dividend_df.columns:
            print(f"Average Dividend Amount: ${dividend_df['Amount'].mean():.4f}")
    
    if not split_df.empty:
        print("\nStock Splits:")
        print(split_df)

def main():
    # Initialize the data fetcher
    fetcher = LeanDataFetcher()
    
    # Print supported exchanges
    print("Supported Exchanges:")
    print(fetcher.get_supported_exchanges())
    
    # Example: Print available symbols for NASDAQ
    print_available_symbols(fetcher, 'NASDAQ')
    
    # Set date range for data fetching
    start_date = datetime.now() - timedelta(days=365)  # Last year
    end_date = datetime.now()
    
    # Example 1: Fetch Apple stock data from NASDAQ with adjusted prices and dividends
    try:
        price_data, dividend_data, split_data = fetcher.fetch_stock_data(
            symbol="AAPL",
            exchange="NASDAQ",
            start_date=start_date,
            end_date=end_date,
            include_dividends=True,
            include_splits=True,
            adjust_prices=True
        )
        
        analyze_stock_data(price_data, dividend_data, split_data, "AAPL")
    
    except Exception as e:
        print(f"Error fetching AAPL data: {str(e)}")
    
    # Example 2: Fetch BHP data from ASX
    try:
        price_data, dividend_data, split_data = fetcher.fetch_stock_data(
            symbol="BHP",
            exchange="ASX",
            start_date=start_date,
            end_date=end_date,
            include_dividends=True,
            include_splits=True,
            adjust_prices=True
        )
        
        analyze_stock_data(price_data, dividend_data, split_data, "BHP")
    
    except Exception as e:
        print(f"Error fetching BHP data: {str(e)}")

if __name__ == "__main__":
    main() 
import yfinance as yf
import pandas as pd
from typing import Dict, Any

def fetch_comprehensive_stock_data(symbol: str = "MSFT") -> Dict[str, Any]:
    """
    Fetch comprehensive stock data for a single symbol using yfinance.
    
    Args:
        symbol: Stock ticker symbol (default: MSFT)
        
    Returns:
        Dictionary containing various stock data metrics
    """
    # Create a Ticker object
    ticker = yf.Ticker(symbol)
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Get company info
        results['info'] = ticker.info
        
        # Get company calendar
        results['calendar'] = ticker.calendar
        
        # Get analyst price targets
        results['analyst_price_targets'] = ticker.analyst_price_targets
        
        # Get quarterly income statement
        results['quarterly_income_stmt'] = ticker.quarterly_income_stmt
        
        # Get historical data (1 month)
        results['historical_data'] = ticker.history(period='1mo')
        
        # Get options data if available
        if len(ticker.options) > 0:
            results['options_data'] = ticker.option_chain(ticker.options[0]).calls
        else:
            results['options_data'] = None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return {}
        
    return results

def display_stock_data(data: Dict[str, Any], symbol: str) -> None:
    """
    Display the fetched stock data in a formatted way.
    
    Args:
        data: Dictionary containing the stock data
        symbol: Stock ticker symbol
    """
    print(f"\n=== Stock Data for {symbol} ===\n")
    
    if not data:
        print("No data available.")
        return
        
    # Display company info (selected fields)
    if 'info' in data and data['info']:
        print("=== Company Info ===")
        info = data['info']
        for key in ['longName', 'sector', 'industry', 'website']:
            if key in info:
                print(f"{key}: {info[key]}")
        print()
    
    # Display calendar
    if 'calendar' in data and data['calendar'] is not None:
        print("=== Calendar ===")
        print(data['calendar'])
        print()
    
    # Display price targets
    if 'analyst_price_targets' in data and data['analyst_price_targets'] is not None:
        print("=== Analyst Price Targets ===")
        print(data['analyst_price_targets'].head())
        print()
    
    # Display quarterly income statement
    if 'quarterly_income_stmt' in data and data['quarterly_income_stmt'] is not None:
        print("=== Quarterly Income Statement ===")
        print(data['quarterly_income_stmt'].head())
        print()
    
    # Display historical data
    if 'historical_data' in data and not data['historical_data'].empty:
        print("=== Historical Data (Last 5 Days) ===")
        print(data['historical_data'].tail())
        print()
    
    # Display options data
    if 'options_data' in data and data['options_data'] is not None:
        print("=== Options Data (First 5 Calls) ===")
        print(data['options_data'].head())
        print()

if __name__ == "__main__":
    # Example usage
    symbol = "MSFT"
    data = fetch_comprehensive_stock_data(symbol)
    display_stock_data(data, symbol)
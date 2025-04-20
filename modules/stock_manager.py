import pandas as pd
import os
from typing import Dict, List, Optional, Union

class StockManager:
    """
    Manager for stock operations.
    Handles stock data, symbols, and metadata.
    """
    
    def __init__(self, symbols_file: str = "stock_symbols.txt"):
        """
        Initialize stock manager.
        
        Args:
            symbols_file (str): Path to file containing stock symbols
        """
        self.symbols_file = symbols_file
        self.symbols = self._load_stock_symbols()
        
    def _load_stock_symbols(self) -> List[str]:
        """
        Load stock symbols from a file, each symbol on a separate line.
        
        Returns:
            List[str]: List of stock symbols
        """
        symbols = []
        try:
            if os.path.exists(self.symbols_file):
                with open(self.symbols_file, "r") as f:
                    for line in f:
                        symbol = line.strip()
                        if symbol:
                            symbols.append(symbol)
            else:
                # Default symbols if file doesn't exist
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        except Exception as e:
            print(f"Error loading stock symbols: {e}")
            # Default symbols on error
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        return symbols
    
    def get_available_stocks(self, search: Optional[str] = None,
                            sector: Optional[str] = None,
                            limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get available stocks with optional filtering.
        
        Args:
            search (str, optional): Search term for stock symbol or name
            sector (str, optional): Filter by sector
            limit (int): Maximum number of stocks to return
            offset (int): Offset for pagination
            
        Returns:
            List[Dict]: List of stock information dictionaries
        """
        # This is a placeholder implementation
        stocks = []
        
        for symbol in self.symbols:
            # Basic stock info
            stock = {
                "symbol": symbol,
                "name": self._get_stock_name(symbol),
                "sector": "Technology",  # Placeholder
                "last_price": 100.0,  # Placeholder
                "last_updated": "2023-01-01T00:00:00Z",  # Placeholder
            }
            
            # Filter by search term if provided
            if search and search.lower() not in symbol.lower() and search.lower() not in stock["name"].lower():
                continue
                
            # Filter by sector if provided
            if sector and stock["sector"].lower() != sector.lower():
                continue
                
            stocks.append(stock)
        
        # Apply pagination
        return stocks[offset:offset+limit]
    
    def get_stock_details(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information for a specific stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Stock details or None if not found
        """
        # Check if symbol exists
        if symbol not in self.symbols:
            return None
            
        # This is a placeholder implementation
        return {
            "symbol": symbol,
            "name": self._get_stock_name(symbol),
            "sector": "Technology",  # Placeholder
            "industry": "Software",  # Placeholder
            "exchange": "NASDAQ",  # Placeholder
            "currency": "USD",
            "country": "US",
            "last_price": 100.0,  # Placeholder
            "price_change": 1.0,  # Placeholder
            "price_change_percent": 1.0,  # Placeholder
            "market_cap": 1000000000.0,  # Placeholder
            "last_updated": "2023-01-01T00:00:00Z",  # Placeholder
        }
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for stocks by symbol or name.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict]: List of matching stocks
        """
        return self.get_available_stocks(search=query, limit=limit)
    
    def _get_stock_name(self, symbol: str) -> str:
        """
        Get the name of a stock by symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            str: Stock name
        """
        # This is a placeholder implementation
        # In a real implementation, this would look up the name from a database
        names = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com, Inc.",
            "META": "Meta Platforms, Inc.",
        }
        return names.get(symbol, f"{symbol} Inc.")


# For backward compatibility with existing code
def load_stock_symbols(filename):
    '''Load stock symbols from a file, each symbol on a separate line.'''
    symbols = []
    try:
        with open(filename, "r") as f:
            for line in f:
                symbol = line.strip()
                if symbol:
                    symbols.append(symbol)
    except Exception as e:
        print(f"Error loading stock symbols: {e}")
    return symbols

def manage_stocks(data):
    """
    Manage stocks based on the provided data.
    This is a placeholder function that would implement custom indicators and risk management.
    """
    print("Managing stocks with data shape:", data.shape)
    # In a real implementation, this would apply custom indicators and risk management
    if isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels > 1:
        stocks = data.columns.get_level_values(1).unique().tolist()
    else:
        stocks = data.columns.tolist()
    return {"status": "success", "managed_stocks": stocks}
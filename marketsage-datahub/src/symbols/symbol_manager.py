#!/usr/bin/env python3
"""
Symbol management module for the MarketSage DataHub component.
Handles stock symbol operations, including loading, saving, and filtering.
"""

import os
import csv
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import pandas as pd
import requests

# Setup logging
logger = logging.getLogger(__name__)


class SymbolManager:
    """
    Manager for stock symbol operations.
    Handles symbol data, filtering, and persistence.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize symbol manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}
        self.symbols_config = self.config.get("symbols", {})
        
        # Get symbols file path from config or use default
        self.symbols_file = self.symbols_config.get("local_file", "data/symbols.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.symbols_file), exist_ok=True)
        
        # Load symbols
        self.symbols = self._load_symbols()
        self.symbol_details = {}
        
        # Check if we should auto-update the symbol list
        if self.symbols_config.get("auto_update", False):
            self._check_update_symbols()
    
    def _load_symbols(self) -> List[str]:
        """
        Load stock symbols from the symbols file.
        
        Returns:
            List[str]: List of stock symbols
        """
        symbols = []
        try:
            if os.path.exists(self.symbols_file):
                if self.symbols_file.endswith('.csv'):
                    # Load from CSV
                    with open(self.symbols_file, "r") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row and row[0].strip():
                                symbols.append(row[0].strip())
                else:
                    # Load from text file (one symbol per line)
                    with open(self.symbols_file, "r") as f:
                        for line in f:
                            symbol = line.strip()
                            if symbol:
                                symbols.append(symbol)
                
                logger.info(f"Loaded {len(symbols)} symbols from {self.symbols_file}")
            else:
                # Default symbols if file doesn't exist
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                logger.warning(f"Symbols file {self.symbols_file} not found, using default symbols")
                
                # Save default symbols
                self._save_symbols(symbols)
        except Exception as e:
            logger.error(f"Error loading stock symbols: {e}")
            # Default symbols on error
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return symbols
    
    def _save_symbols(self, symbols: List[str]) -> bool:
        """
        Save stock symbols to the symbols file.
        
        Args:
            symbols (List[str]): List of stock symbols
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.symbols_file), exist_ok=True)
            
            if self.symbols_file.endswith('.csv'):
                # Save as CSV
                with open(self.symbols_file, "w", newline='') as f:
                    writer = csv.writer(f)
                    for symbol in symbols:
                        writer.writerow([symbol])
            else:
                # Save as text file (one symbol per line)
                with open(self.symbols_file, "w") as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
            
            logger.info(f"Saved {len(symbols)} symbols to {self.symbols_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving stock symbols: {e}")
            return False
    
    def _check_update_symbols(self) -> None:
        """Check if symbols list needs updating and update if necessary."""
        try:
            # Check if file exists and when it was last modified
            if os.path.exists(self.symbols_file):
                last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(self.symbols_file))
                now = datetime.datetime.now()
                
                # Get update frequency from config (in days)
                update_frequency = self.symbols_config.get("update_frequency", 7)
                
                # Check if update is needed
                if (now - last_modified).days < update_frequency:
                    logger.info(f"Symbol list is up to date (last updated {last_modified})")
                    return
            
            logger.info("Updating symbol list...")
            self._update_symbols()
            
        except Exception as e:
            logger.error(f"Error checking symbol list update: {e}")
    
    def _update_symbols(self) -> bool:
        """
        Update the symbols list from external sources.
        
        Returns:
            bool: Success status
        """
        # This is a placeholder for symbol list updating
        # In a real implementation, this would fetch symbols from an API
        logger.info("Symbol list update not implemented yet")
        return False
    
    def get_symbols(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get available stock symbols with optional filtering.
        
        Args:
            filters (Dict[str, Any], optional): Filtering criteria
            
        Returns:
            List[str]: List of stock symbols
        """
        if not filters:
            return self.symbols
        
        # Apply filters
        filtered_symbols = []
        for symbol in self.symbols:
            # Get symbol details if available
            details = self.get_symbol_details(symbol)
            
            # Check if symbol matches filters
            if self._matches_filters(symbol, details, filters):
                filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def _matches_filters(self, symbol: str, details: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a symbol matches the specified filters.
        
        Args:
            symbol (str): Stock symbol
            details (Dict[str, Any]): Symbol details
            filters (Dict[str, Any]): Filtering criteria
            
        Returns:
            bool: True if the symbol matches the filters
        """
        # Check exchange filter
        if "exchanges" in filters and details.get("exchange"):
            if details["exchange"] not in filters["exchanges"]:
                return False
        
        # Check market cap filter
        if "min_market_cap" in filters and details.get("market_cap"):
            if details["market_cap"] < filters["min_market_cap"]:
                return False
        
        # Check sector filter
        if "sector" in filters and details.get("sector"):
            if details["sector"] != filters["sector"]:
                return False
        
        # Check industry filter
        if "industry" in filters and details.get("industry"):
            if details["industry"] != filters["industry"]:
                return False
        
        return True
    
    def add_symbols(self, symbols: List[str]) -> List[str]:
        """
        Add symbols to the managed list.
        
        Args:
            symbols (List[str]): List of symbols to add
            
        Returns:
            List[str]: List of symbols that were added
        """
        added = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol and symbol not in self.symbols:
                self.symbols.append(symbol)
                added.append(symbol)
        
        # Save updated symbols list
        if added:
            self._save_symbols(self.symbols)
            logger.info(f"Added {len(added)} symbols")
        
        return added
    
    def remove_symbols(self, symbols: List[str]) -> List[str]:
        """
        Remove symbols from the managed list.
        
        Args:
            symbols (List[str]): List of symbols to remove
            
        Returns:
            List[str]: List of symbols that were removed
        """
        removed = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                removed.append(symbol)
        
        # Save updated symbols list
        if removed:
            self._save_symbols(self.symbols)
            logger.info(f"Removed {len(removed)} symbols")
        
        return removed
    
    def get_symbol_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Symbol details
        """
        # Check if we already have details for this symbol
        if symbol in self.symbol_details:
            return self.symbol_details[symbol]
        
        # This is a placeholder implementation
        # In a real implementation, this would fetch details from an API
        details = {
            "symbol": symbol,
            "name": self._get_symbol_name(symbol),
            "sector": "Technology",  # Placeholder
            "industry": "Software",  # Placeholder
            "exchange": "NASDAQ",  # Placeholder
            "currency": "USD",
            "country": "US",
            "market_cap": 1000000000.0,  # Placeholder
        }
        
        # Cache the details
        self.symbol_details[symbol] = details
        
        return details
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for symbols by symbol or name.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching symbol information
        """
        results = []
        query = query.lower()
        
        for symbol in self.symbols:
            # Get symbol details
            details = self.get_symbol_details(symbol)
            
            # Check if symbol or name matches query
            if query in symbol.lower() or query in details.get("name", "").lower():
                results.append(details)
                
                # Check limit
                if len(results) >= limit:
                    break
        
        return results
    
    def _get_symbol_name(self, symbol: str) -> str:
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
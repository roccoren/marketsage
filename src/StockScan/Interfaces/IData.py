"""
IData.py - Data Interface

This module defines the standard interface for market data providers and handlers.
Following LEAN best practices, this ensures consistency across different data implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IData(ABC):
    """
    Interface for all market data providers and handlers.
    Following LEAN framework best practices.
    """
    
    @abstractmethod
    def Initialize(self) -> None:
        """
        Initialize the data provider.
        """
        pass
    
    @abstractmethod
    async def GetHistory(self, 
                   symbol: str, 
                   start_date: datetime, 
                   end_date: datetime,
                   resolution: str = "Daily") -> Any:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            start_date: The start date
            end_date: The end date
            resolution: The data resolution
            
        Returns:
            The historical data
        """
        pass
    
    @abstractmethod
    async def GetLatestData(self, symbol: str, resolution: str = "Daily") -> Any:
        """
        Get the latest data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            resolution: The data resolution
            
        Returns:
            The latest data
        """
        pass
    
    @abstractmethod
    async def GetMultipleSymbols(self, 
                          symbols: List[str], 
                          start_date: datetime,
                          end_date: datetime,
                          resolution: str = "Daily") -> Dict[str, Any]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: The list of symbols to get data for
            start_date: The start date
            end_date: The end date
            resolution: The data resolution
            
        Returns:
            Dictionary mapping symbols to their historical data
        """
        pass
    
    @abstractmethod
    async def FetchMarketData(self, 
                        market: str,
                        symbols: Optional[List[str]] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        resolution: str = "Daily") -> None:
        """
        Fetch market data for the specified market and store it.
        
        Args:
            market: The market to fetch data for (e.g., nyse, nasdaq)
            symbols: Optional list of specific symbols to fetch
            start_date: Optional start date
            end_date: Optional end date
            resolution: The data resolution
        """
        pass
    
    @abstractmethod
    def ConvertToLeanFormat(self, data: Any, symbol: str, output_path: Optional[str] = None) -> str:
        """
        Convert data to LEAN format.
        
        Args:
            data: The data to convert
            symbol: The symbol
            output_path: Optional output path
            
        Returns:
            The path to the converted data
        """
        pass


class IDataProvider(ABC):
    """
    Interface for market data providers.
    """
    
    @abstractmethod
    async def FetchData(self, 
                  symbol: str, 
                  start_date: datetime,
                  end_date: datetime,
                  resolution: str = "Daily") -> Any:
        """
        Fetch data for a symbol.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: The start date
            end_date: The end date
            resolution: The data resolution
            
        Returns:
            The fetched data
        """
        pass
    
    @abstractmethod
    async def FetchMultipleSymbols(self, 
                             symbols: List[str],
                             start_date: datetime,
                             end_date: datetime,
                             resolution: str = "Daily") -> Dict[str, Any]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: The list of symbols to fetch data for
            start_date: The start date
            end_date: The end date
            resolution: The data resolution
            
        Returns:
            Dictionary mapping symbols to their data
        """
        pass 
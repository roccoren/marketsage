"""
IAlgorithm.py - Algorithm Interface

This module defines the standard interface that all trading algorithms must implement.
Following LEAN best practices, this ensures consistency across different algorithm implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IAlgorithm(ABC):
    """
    Interface that all trading algorithms must implement.
    Following LEAN framework best practices.
    """
    
    @abstractmethod
    def Initialize(self) -> None:
        """
        Initialize the algorithm settings and state. 
        This is called once at the start of the algorithm.
        """
        pass
    
    @abstractmethod
    def OnData(self, data: Any) -> None:
        """
        Event handler for market data updates.
        This is the main entry point for trading logic.
        
        Args:
            data: The new data available
        """
        pass
    
    @abstractmethod
    def OnOrderEvent(self, order_event: Any) -> None:
        """
        Event handler for order events (fills, cancellations, etc.)
        
        Args:
            order_event: The order event information
        """
        pass
    
    @abstractmethod
    def OnEndOfDay(self, symbol: Any = None) -> None:
        """
        Event handler called at the end of each trading day.
        
        Args:
            symbol: The symbol for which the day has ended, or None for all symbols
        """
        pass
    
    @abstractmethod
    def OnEndOfAlgorithm(self) -> None:
        """
        Event handler called at the end of the algorithm execution.
        Use this for final cleanup and reporting.
        """
        pass
    
    @abstractmethod
    def SetStartDate(self, year: int, month: int, day: int) -> None:
        """
        Set the start date for the algorithm.
        
        Args:
            year: The year
            month: The month
            day: The day
        """
        pass
    
    @abstractmethod
    def SetEndDate(self, year: int, month: int, day: int) -> None:
        """
        Set the end date for the algorithm.
        
        Args:
            year: The year
            month: The month
            day: The day
        """
        pass
    
    @abstractmethod
    def SetCash(self, amount: float) -> None:
        """
        Set the starting cash amount for the algorithm.
        
        Args:
            amount: The cash amount
        """
        pass
    
    @abstractmethod
    def AddEquity(self, symbol: str, resolution: Any) -> Any:
        """
        Add an equity to the algorithm universe.
        
        Args:
            symbol: The equity symbol
            resolution: The data resolution
            
        Returns:
            The security object
        """
        pass
    
    @abstractmethod
    def SetBrokerageModel(self, brokerage_model: Any) -> None:
        """
        Set the brokerage model for the algorithm.
        
        Args:
            brokerage_model: The brokerage model
        """
        pass 
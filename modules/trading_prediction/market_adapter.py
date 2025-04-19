#!/usr/bin/env python3
"""
Market adapter module for handling different market-specific behaviors.

This module provides adapters for different stock markets (US, Chinese, etc.),
handling market hours, trading calendars, and market-specific rules.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import pytz
from holidays import country_holidays

from modules.trading_prediction.base import MarketAdapter

# Configure logger
logger = logging.getLogger(__name__)


class BaseMarketAdapter(MarketAdapter):
    """Base implementation for market adapters."""
    
    def __init__(self, market_name: str, timezone: str):
        """
        Initialize BaseMarketAdapter.
        
        Args:
            market_name: Name of the market
            timezone: Timezone for the market (e.g., 'America/New_York')
        """
        self._market_name = market_name
        self._timezone = pytz.timezone(timezone)
        self._holidays = {}
        
    @property
    def name(self) -> str:
        """Get market name."""
        return self._market_name
    
    @property
    def timezone(self) -> pytz.timezone:
        """Get market timezone."""
        return self._timezone
    
    def _is_weekend(self, date: datetime) -> bool:
        """
        Check if date is a weekend.
        
        Args:
            date: Date to check
            
        Returns:
            True if weekend, False otherwise
        """
        return date.weekday() >= 5  # 5 = Saturday, 6 = Sunday
    
    def _load_holidays(self, country_code: str, years: List[int]) -> Dict[datetime, str]:
        """
        Load holidays for a country.
        
        Args:
            country_code: ISO country code
            years: List of years to load holidays for
            
        Returns:
            Dictionary mapping dates to holiday names
        """
        holidays = {}
        for year in years:
            yearly_holidays = country_holidays(country_code, years=year)
            holidays.update(yearly_holidays)
        return holidays
    
    def _convert_to_datetime(self, date_str: str) -> datetime:
        """
        Convert string date to datetime.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            datetime object
        """
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    def _convert_to_string(self, date: datetime) -> str:
        """
        Convert datetime to string.
        
        Args:
            date: datetime object
            
        Returns:
            Date in YYYY-MM-DD format
        """
        return date.strftime('%Y-%m-%d')
    
    def is_trading_day(self, date: str) -> bool:
        """
        Check if a given date is a trading day.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            True if trading day, False otherwise
        """
        dt = self._convert_to_datetime(date)
        # Not a trading day if weekend
        if self._is_weekend(dt):
            return False
        
        # Not a trading day if holiday
        if dt in self._holidays:
            return False
        
        return True
    
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of trading days for this market.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of trading days in YYYY-MM-DD format
        """
        start_dt = self._convert_to_datetime(start_date)
        end_dt = self._convert_to_datetime(end_date)
        
        # Generate all dates in range
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if self.is_trading_day(current_dt):
                dates.append(self._convert_to_string(current_dt))
            current_dt += timedelta(days=1)
        
        return dates
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Default implementation for symbol normalization.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Normalized symbol
        """
        return symbol.upper()


class USMarketAdapter(BaseMarketAdapter):
    """Adapter for US stock markets (NYSE, NASDAQ)."""
    
    def __init__(self):
        """Initialize US market adapter."""
        super().__init__("US", "America/New_York")
        
        # Regular trading hours (in market's timezone)
        self._regular_hours = {
            "start": time(9, 30),  # 9:30 AM ET
            "end": time(16, 0)     # 4:00 PM ET
        }
        
        # Pre-market hours
        self._pre_market_hours = {
            "start": time(4, 0),   # 4:00 AM ET
            "end": time(9, 30)     # 9:30 AM ET
        }
        
        # After-hours
        self._after_hours = {
            "start": time(16, 0),  # 4:00 PM ET
            "end": time(20, 0)     # 8:00 PM ET
        }
        
        # Load US holidays for current year and next year
        current_year = datetime.now().year
        self._holidays = self._load_holidays('US', [current_year - 1, current_year, current_year + 1])
        
        # Add special closings (half days, etc.)
        self._special_closings = self._load_special_closings()
    
    def _load_special_closings(self) -> Dict[datetime, time]:
        """
        Load special market closings (e.g., early closures).
        
        Returns:
            Dictionary mapping dates to early closing times
        """
        # Example special closings - in a real implementation, this would be loaded from a data source
        special_closings = {
            # Format: datetime(year, month, day): time(hour, minute)
            datetime(2024, 7, 3): time(13, 0),  # Early close before Independence Day
            datetime(2024, 11, 29): time(13, 0),  # Early close after Thanksgiving
            datetime(2024, 12, 24): time(13, 0),  # Christmas Eve
        }
        return special_closings
    
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours information.
        
        Returns:
            Dictionary with market hours information
        """
        return {
            "timezone": self._timezone.zone,
            "regular_hours": self._regular_hours,
            "pre_market_hours": self._pre_market_hours,
            "after_hours": self._after_hours
        }
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """
        Check if market is open at the given datetime.
        
        Args:
            dt: datetime to check (default: current time)
            
        Returns:
            True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now(self._timezone)
        else:
            # Ensure datetime is timezone-aware
            if dt.tzinfo is None:
                dt = self._timezone.localize(dt)
            else:
                dt = dt.astimezone(self._timezone)
        
        # Not open if not a trading day
        if not self.is_trading_day(dt):
            return False
        
        # Check if it's a special closing day
        dt_date = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if dt_date in self._special_closings:
            closing_time = self._special_closings[dt_date]
            return dt.time() >= self._regular_hours["start"] and dt.time() < closing_time
        
        # Regular hours check
        return dt.time() >= self._regular_hours["start"] and dt.time() < self._regular_hours["end"]
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for US markets.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Normalized symbol
        """
        # US symbols are typically uppercase without any special formatting
        return symbol.upper()


class ChineseMarketAdapter(BaseMarketAdapter):
    """Adapter for Chinese stock markets (SSE, SZSE)."""
    
    def __init__(self):
        """Initialize Chinese market adapter."""
        super().__init__("China", "Asia/Shanghai")
        
        # Regular trading hours (in market's timezone)
        # Chinese markets have a morning and afternoon session
        self._morning_session = {
            "start": time(9, 30),  # 9:30 AM CST
            "end": time(11, 30)    # 11:30 AM CST
        }
        
        self._afternoon_session = {
            "start": time(13, 0),  # 1:00 PM CST
            "end": time(15, 0)     # 3:00 PM CST
        }
        
        # Load Chinese holidays for current year and next year
        current_year = datetime.now().year
        self._holidays = self._load_chinese_holidays([current_year - 1, current_year, current_year + 1])
    
    def _load_chinese_holidays(self, years: List[int]) -> Dict[datetime, str]:
        """
        Load Chinese holidays, which include both Western and Lunar calendar holidays.
        
        Args:
            years: List of years to load holidays for
            
        Returns:
            Dictionary mapping dates to holiday names
        """
        # In a real implementation, this would load from a proper Chinese holiday calendar
        # For now, we'll use a simplified approach with some common Chinese holidays
        
        holidays = {}
        
        # Common Chinese holidays (these dates change each year based on lunar calendar)
        # This is just an example - a real implementation would use a proper Chinese calendar
        chinese_holidays = {
            # Format: (year, month, day): "Holiday Name"
            (2024, 1, 1): "New Year's Day",
            (2024, 2, 10): "Chinese New Year",
            (2024, 2, 11): "Chinese New Year",
            (2024, 2, 12): "Chinese New Year",
            (2024, 2, 13): "Chinese New Year",
            (2024, 2, 14): "Chinese New Year",
            (2024, 2, 15): "Chinese New Year",
            (2024, 2, 16): "Chinese New Year",
            (2024, 4, 4): "Qingming Festival",
            (2024, 5, 1): "Labor Day",
            (2024, 5, 2): "Labor Day",
            (2024, 5, 3): "Labor Day",
            (2024, 6, 10): "Dragon Boat Festival",
            (2024, 9, 17): "Mid-Autumn Festival",
            (2024, 10, 1): "National Day",
            (2024, 10, 2): "National Day",
            (2024, 10, 3): "National Day",
            (2024, 10, 4): "National Day",
            (2024, 10, 5): "National Day",
            (2024, 10, 6): "National Day",
            (2024, 10, 7): "National Day",
        }
        
        for (year, month, day), holiday_name in chinese_holidays.items():
            if year in years:
                holidays[datetime(year, month, day)] = holiday_name
        
        return holidays
    
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours information.
        
        Returns:
            Dictionary with market hours information
        """
        return {
            "timezone": self._timezone.zone,
            "morning_session": self._morning_session,
            "afternoon_session": self._afternoon_session
        }
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """
        Check if market is open at the given datetime.
        
        Args:
            dt: datetime to check (default: current time)
            
        Returns:
            True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now(self._timezone)
        else:
            # Ensure datetime is timezone-aware
            if dt.tzinfo is None:
                dt = self._timezone.localize(dt)
            else:
                dt = dt.astimezone(self._timezone)
        
        # Not open if not a trading day
        if not self.is_trading_day(dt):
            return False
        
        current_time = dt.time()
        
        # Check if in morning session
        in_morning = (current_time >= self._morning_session["start"] and 
                     current_time < self._morning_session["end"])
        
        # Check if in afternoon session
        in_afternoon = (current_time >= self._afternoon_session["start"] and 
                       current_time < self._afternoon_session["end"])
        
        return in_morning or in_afternoon
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for Chinese markets.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Normalized symbol
        """
        # Chinese symbols typically have a market prefix (e.g., 'SH' for Shanghai, 'SZ' for Shenzhen)
        symbol = symbol.upper()
        
        # If symbol doesn't have a market prefix, try to determine it
        if not (symbol.startswith('SH') or symbol.startswith('SZ')):
            # Simple heuristic: 6xxxxx is typically Shanghai, 0xxxxx or 3xxxxx is typically Shenzhen
            if symbol.isdigit():
                if symbol.startswith('6'):
                    symbol = f"SH{symbol}"
                elif symbol.startswith('0') or symbol.startswith('3'):
                    symbol = f"SZ{symbol}"
        
        return symbol


class MarketAdapterFactory:
    """Factory for creating market adapters."""
    
    @staticmethod
    def create_adapter(market: str) -> MarketAdapter:
        """
        Create a market adapter for the specified market.
        
        Args:
            market: Market identifier ('US', 'China', etc.)
            
        Returns:
            MarketAdapter instance
            
        Raises:
            ValueError: If market is not supported
        """
        market = market.lower()
        
        if market in ['us', 'nyse', 'nasdaq']:
            return USMarketAdapter()
        elif market in ['china', 'sse', 'szse']:
            return ChineseMarketAdapter()
        else:
            raise ValueError(f"Unsupported market: {market}")


class MarketCalendarManager:
    """Manager for handling multiple market calendars."""
    
    def __init__(self):
        """Initialize MarketCalendarManager."""
        self._adapters = {}
    
    def register_adapter(self, adapter: MarketAdapter) -> None:
        """
        Register a market adapter.
        
        Args:
            adapter: MarketAdapter instance
        """
        self._adapters[adapter.name] = adapter
    
    def get_adapter(self, market: str) -> MarketAdapter:
        """
        Get adapter for a specific market.
        
        Args:
            market: Market name
            
        Returns:
            MarketAdapter instance
            
        Raises:
            KeyError: If market adapter is not registered
        """
        if market not in self._adapters:
            # Try to create adapter on demand
            try:
                adapter = MarketAdapterFactory.create_adapter(market)
                self.register_adapter(adapter)
                return adapter
            except ValueError:
                raise KeyError(f"No adapter registered for market: {market}")
        
        return self._adapters[market]
    
    def get_common_trading_days(self, markets: List[str], start_date: str, end_date: str) -> List[str]:
        """
        Get trading days common to all specified markets.
        
        Args:
            markets: List of market names
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of common trading days in YYYY-MM-DD format
        """
        if not markets:
            return []
        
        # Get trading days for each market
        market_trading_days = {}
        for market in markets:
            adapter = self.get_adapter(market)
            market_trading_days[market] = set(adapter.get_trading_days(start_date, end_date))
        
        # Find common trading days
        common_days = set.intersection(*market_trading_days.values())
        
        # Convert back to list and sort
        return sorted(list(common_days))
    
    def convert_datetime(self, dt: datetime, from_market: str, to_market: str) -> datetime:
        """
        Convert datetime from one market's timezone to another.
        
        Args:
            dt: datetime to convert
            from_market: Source market
            to_market: Target market
            
        Returns:
            Converted datetime
        """
        from_adapter = self.get_adapter(from_market)
        to_adapter = self.get_adapter(to_market)
        
        # Ensure datetime is timezone-aware in source timezone
        if dt.tzinfo is None:
            dt = from_adapter.timezone.localize(dt)
        else:
            dt = dt.astimezone(from_adapter.timezone)
        
        # Convert to target timezone
        return dt.astimezone(to_adapter.timezone)
    
    def get_next_trading_day(self, market: str, date: str) -> str:
        """
        Get the next trading day after the specified date.
        
        Args:
            market: Market name
            date: Date in YYYY-MM-DD format
            
        Returns:
            Next trading day in YYYY-MM-DD format
        """
        adapter = self.get_adapter(market)
        dt = adapter._convert_to_datetime(date)
        
        # Increment by one day until we find a trading day
        dt += timedelta(days=1)
        while not adapter.is_trading_day(dt):
            dt += timedelta(days=1)
        
        return adapter._convert_to_string(dt)
    
    def get_previous_trading_day(self, market: str, date: str) -> str:
        """
        Get the previous trading day before the specified date.
        
        Args:
            market: Market name
            date: Date in YYYY-MM-DD format
            
        Returns:
            Previous trading day in YYYY-MM-DD format
        """
        adapter = self.get_adapter(market)
        dt = adapter._convert_to_datetime(date)
        
        # Decrement by one day until we find a trading day
        dt -= timedelta(days=1)
        while not adapter.is_trading_day(dt):
            dt -= timedelta(days=1)
        
        return adapter._convert_to_string(dt)


# Create a global instance of the market calendar manager
_market_calendar_manager = None

def get_market_calendar_manager() -> MarketCalendarManager:
    """
    Get the global market calendar manager instance.
    
    Returns:
        MarketCalendarManager instance
    """
    global _market_calendar_manager
    
    if _market_calendar_manager is None:
        _market_calendar_manager = MarketCalendarManager()
        
        # Register default adapters
        _market_calendar_manager.register_adapter(USMarketAdapter())
        _market_calendar_manager.register_adapter(ChineseMarketAdapter())
    
    return _market_calendar_manager
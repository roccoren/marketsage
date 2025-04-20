#!/usr/bin/env python3
"""
Stock schemas for the StockScan API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class StockDataPoint(BaseModel):
    """Stock data point schema."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: Optional[Dict[str, float]] = None


class StockData(BaseModel):
    """Stock data schema."""
    
    symbol: str
    name: Optional[str] = None
    data: List[StockDataPoint]
    last_updated: datetime


class StockInfo(BaseModel):
    """Stock information schema."""
    
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = "USD"
    country: Optional[str] = None
    last_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_percent: Optional[float] = None
    market_cap: Optional[float] = None
    last_updated: Optional[datetime] = None


class StockList(BaseModel):
    """Stock list schema."""
    
    stocks: List[StockInfo]
    total: int


class StockDataRequest(BaseModel):
    """Stock data request schema."""
    
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"  # 1d, 5m, 1h, etc.
    
    @validator("interval")
    def validate_interval(cls, v):
        """Validate interval."""
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval: {v}. Must be one of {valid_intervals}")
        return v


class StockRefreshRequest(BaseModel):
    """Stock refresh request schema."""
    
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"
    
    @validator("interval")
    def validate_interval(cls, v):
        """Validate interval."""
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval: {v}. Must be one of {valid_intervals}")
        return v


class StockRefreshResponse(BaseModel):
    """Stock refresh response schema."""
    
    symbols: List[str]
    status: str
    message: str
    refresh_id: Optional[str] = None
    job_status: Optional[str] = None


class StockRefreshStatus(BaseModel):
    """Stock refresh status schema."""
    
    refresh_id: str
    status: str
    message: str
    symbols: List[str]
    completed: List[str] = []
    failed: Dict[str, str] = {}
    progress: float = 0.0
    start_time: datetime
    end_time: Optional[datetime] = None


class StockSearchRequest(BaseModel):
    """Stock search request schema."""
    
    query: str
    limit: Optional[int] = 10


class StockSearchResponse(BaseModel):
    """Stock search response schema."""
    
    results: List[StockInfo]
    total: int


class StockIndicator(BaseModel):
    """Stock indicator schema."""
    
    name: str
    value: float
    signal: Optional[str] = None  # buy, sell, hold
    description: Optional[str] = None


class StockAnalysis(BaseModel):
    """Stock analysis schema."""
    
    symbol: str
    name: Optional[str] = None
    last_price: float
    price_change: float
    price_change_percent: float
    indicators: Dict[str, StockIndicator]
    analysis_date: datetime
    summary: str
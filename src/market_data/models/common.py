"""Common data models used across the market data package."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

@dataclass
class MarketData:
    """Represents a single point of market data."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None

@dataclass
class TickerInfo:
    """Basic information about a ticker/symbol."""
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

@dataclass
class HistoricalData:
    """Container for historical market data."""
    symbol: str
    data: List[MarketData]
    start_date: datetime
    end_date: datetime
    interval: str = "1d"  # Default to daily data
    meta: Optional[Dict[str, Any]] = None

@dataclass
class RequestParams:
    """Parameters for data requests."""
    symbols: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"
    adjust_prices: bool = True
    include_meta: bool = False

@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    enabled: bool = True
    ttl: int = 3600  # Default 1 hour TTL
    max_size: Optional[int] = None
    directory: Optional[str] = None

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 0.5  # Default to 2 requests per second
    burst_size: int = 10
    retry_after: int = 60  # Seconds to wait after hitting rate limit

@dataclass
class ProviderConfig:
    """Configuration for data provider."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
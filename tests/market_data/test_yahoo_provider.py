"""Tests for Yahoo Finance market data provider."""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.market_data.providers.yahoo import YahooFinanceProvider
from src.market_data.models.common import (
    ProviderConfig,
    RateLimitConfig,
    CacheConfig,
    MarketData,
    TickerInfo
)
from src.market_data.exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError
)

@pytest.fixture
def mock_yfinance():
    """Create mock yfinance module."""
    with patch('src.market_data.providers.yahoo.yf') as mock_yf:
        yield mock_yf

@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return ProviderConfig(
        name="yahoo_test",
        rate_limit=RateLimitConfig(
            requests_per_second=2.0,
            burst_size=3
        ),
        cache=CacheConfig(
            enabled=True,
            directory="./test_cache"
        )
    )

@pytest.fixture
def yahoo_provider(provider_config):
    """Create Yahoo Finance provider instance."""
    return YahooFinanceProvider(provider_config)

def create_mock_history_df():
    """Create mock pandas DataFrame with test data."""
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-05',
        freq='D'
    )
    
    data = {
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }
    
    return pd.DataFrame(data, index=dates)

@pytest.mark.asyncio
async def test_fetch_history_basic(mock_yfinance, yahoo_provider):
    """Test basic historical data fetching."""
    # Setup mock
    mock_ticker = Mock()
    mock_ticker.history.return_value = create_mock_history_df()
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Fetch data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    
    result = await yahoo_provider._fetch_history("AAPL", start_date, end_date)
    
    # Verify results
    assert len(result) == 5
    assert all(isinstance(item, MarketData) for item in result)
    assert result[0].date.date() == start_date.date()
    assert result[-1].date.date() == (end_date - timedelta(days=1)).date()

@pytest.mark.asyncio
async def test_fetch_history_empty_response(mock_yfinance, yahoo_provider):
    """Test handling of empty response."""
    # Setup mock
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Attempt to fetch data
    with pytest.raises(DataNotFoundError):
        await yahoo_provider._fetch_history(
            "INVALID",
            datetime(2024, 1, 1),
            datetime(2024, 1, 5)
        )

@pytest.mark.asyncio
async def test_fetch_history_connection_error(mock_yfinance, yahoo_provider):
    """Test handling of connection errors."""
    # Setup mock
    mock_ticker = Mock()
    mock_ticker.history.side_effect = Exception(
        "Failed to establish a new connection"
    )
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Attempt to fetch data
    with pytest.raises(ConnectionError):
        await yahoo_provider._fetch_history(
            "AAPL",
            datetime(2024, 1, 1),
            datetime(2024, 1, 5)
        )

@pytest.mark.asyncio
async def test_fetch_ticker_info(mock_yfinance, yahoo_provider):
    """Test fetching ticker information."""
    # Setup mock
    mock_info = {
        'longName': 'Apple Inc.',
        'exchange': 'NASDAQ',
        'currency': 'USD',
        'marketCap': 2000000000000,
        'sector': 'Technology',
        'industry': 'Consumer Electronics'
    }
    
    mock_ticker = Mock()
    mock_ticker.info = mock_info
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Fetch info
    result = await yahoo_provider._fetch_ticker_info("AAPL")
    
    # Verify results
    assert isinstance(result, TickerInfo)
    assert result.name == 'Apple Inc.'
    assert result.exchange == 'NASDAQ'
    assert result.market_cap == 2000000000000

@pytest.mark.asyncio
async def test_batch_history_fetching(mock_yfinance, yahoo_provider):
    """Test fetching historical data for multiple symbols."""
    # Setup mock
    mock_df = create_mock_history_df()
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_df
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Fetch data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    
    results = await yahoo_provider.get_history_batch(symbols, start_date, end_date)
    
    # Verify results
    assert len(results) == 3
    for symbol in symbols:
        assert symbol in results
        assert results[symbol] is not None
        assert len(results[symbol].data) == 5

@pytest.mark.asyncio
async def test_rate_limiting(mock_yfinance, yahoo_provider):
    """Test rate limiting during data fetching."""
    # Setup mock
    mock_ticker = Mock()
    mock_ticker.history.return_value = create_mock_history_df()
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Make multiple requests quickly
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    
    # Should allow burst_size requests
    for _ in range(yahoo_provider.rate_limiter.max_tokens):
        await yahoo_provider.get_history("AAPL", start_date, end_date)
    
    # Next request should be rate limited and wait
    start_time = time.time()
    await yahoo_provider.get_history("AAPL", start_date, end_date)
    elapsed = time.time() - start_time
    
    assert elapsed >= 0.5  # Should have waited at least 500ms

@pytest.mark.asyncio
async def test_invalid_dates(yahoo_provider):
    """Test validation of invalid date ranges."""
    with pytest.raises(ValidationError):
        await yahoo_provider.get_history(
            "AAPL",
            datetime(2024, 1, 5),  # End date
            datetime(2024, 1, 1)   # Start date (invalid)
        )

@pytest.mark.asyncio
async def test_invalid_symbols(yahoo_provider):
    """Test validation of invalid symbols."""
    invalid_symbols = ["", None, 123, "$$$"]
    
    for symbol in invalid_symbols:
        with pytest.raises(ValidationError):
            await yahoo_provider.get_history(
                symbol,
                datetime(2024, 1, 1),
                datetime(2024, 1, 5)
            )
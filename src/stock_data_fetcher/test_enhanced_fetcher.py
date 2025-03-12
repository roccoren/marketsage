import asyncio
import json
import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from .enhanced_market_fetcher import MarketRegistry, EnhancedMarketDataFetcher
from market_data.models.common import MarketData

@pytest.fixture
def market_registry(tmp_path):
    """Create a test market registry with sample data."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create test market listings
    nasdaq_symbols = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"}
    ]
    
    nyse_symbols = [
        {"symbol": "IBM", "name": "IBM Corp."},
        {"symbol": "GE", "name": "General Electric Co."}
    ]
    
    with open(config_dir / "nasdaq-listed.json", "w") as f:
        json.dump(nasdaq_symbols, f)
    with open(config_dir / "nyse-listed.json", "w") as f:
        json.dump(nyse_symbols, f)
    
    return MarketRegistry(str(config_dir))

@pytest.fixture
def market_data_fetcher(tmp_path):
    """Create a test market data fetcher."""
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    
    # Create test market listings
    with open(config_dir / "nasdaq-listed.json", "w") as f:
        json.dump([{"symbol": "AAPL", "name": "Apple Inc."}], f)
    
    return EnhancedMarketDataFetcher(
        data_dir=str(data_dir),
        config_dir=str(config_dir)
    )

def test_market_registry_initialization(market_registry):
    """Test market registry loads markets correctly."""
    markets = market_registry.get_markets()
    assert "nasdaq" in markets
    assert "nyse" in markets
    
    nasdaq_symbols = market_registry.get_symbols("nasdaq")
    assert "AAPL" in nasdaq_symbols
    assert "MSFT" in nasdaq_symbols
    
    nyse_symbols = market_registry.get_symbols("nyse")
    assert "IBM" in nyse_symbols
    assert "GE" in nyse_symbols

def test_market_registry_validation(market_registry):
    """Test symbol validation."""
    assert market_registry.validate_symbol("nasdaq", "AAPL") is True
    assert market_registry.validate_symbol("nasdaq", "INVALID") is False
    assert market_registry.validate_symbol("invalid_market", "AAPL") is False

@pytest.mark.asyncio
async def test_get_existing_dates(market_data_fetcher, tmp_path):
    """Test detection of existing data files."""
    # Create some test data files
    market = "nasdaq"
    symbol = "AAPL"
    symbol_dir = tmp_path / "data" / market / symbol
    symbol_dir.mkdir(parents=True)
    
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3)
    ]
    
    for date in dates:
        file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(file_path, "w") as f:
            json.dump({"close": 100.0}, f)
    
    existing_dates = await market_data_fetcher._get_existing_dates(market, symbol)
    assert len(existing_dates) == 3
    assert all(date in existing_dates for date in dates)

@pytest.mark.asyncio
async def test_backfill_data(market_data_fetcher):
    """Test data backfilling functionality."""
    market = "nasdaq"
    symbol = "AAPL"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    
    # Mock provider response
    mock_data = Mock()
    mock_data.date = start_date
    mock_data.open = 100.0
    mock_data.high = 105.0
    mock_data.low = 95.0
    mock_data.close = 102.0
    mock_data.volume = 1000000.0
    mock_data.dividends = 0.0
    mock_data.stock_splits = 0.0
    
    mock_result = Mock()
    mock_result.data = [mock_data]
    
    with patch.object(market_data_fetcher.provider, 'get_history', 
                     return_value=mock_result):
        await market_data_fetcher.backfill_data(
            market,
            symbol,
            start_date,
            end_date
        )
        
        # Verify data was saved
        symbol_dir = Path(market_data_fetcher.data_dir) / market / symbol
        saved_files = list(symbol_dir.glob("*.json"))
        assert len(saved_files) > 0
        
        # Verify file content
        with open(saved_files[0]) as f:
            data = json.load(f)
            assert data["close"] == 102.0
            assert data["volume"] == 1000000.0

@pytest.mark.asyncio
async def test_fetch_market_data(market_data_fetcher):
    """Test fetching data for entire market."""
    market = "nasdaq"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    # Mock backfill_data to track calls
    with patch.object(market_data_fetcher, 'backfill_data') as mock_backfill:
        await market_data_fetcher.fetch_market_data(
            market,
            start_date,
            end_date
        )
        
        # Verify backfill was called for each symbol
        symbols = market_data_fetcher.registry.get_symbols(market)
        assert mock_backfill.call_count == len(symbols)

def test_invalid_market(market_data_fetcher):
    """Test handling of invalid market."""
    with pytest.raises(ValueError):
        asyncio.run(market_data_fetcher.fetch_market_data(
            "invalid_market",
            datetime.now()
        ))

def test_invalid_symbol(market_data_fetcher):
    """Test handling of invalid symbol."""
    with pytest.raises(ValueError):
        asyncio.run(market_data_fetcher.backfill_data(
            "nasdaq",
            "INVALID",
            datetime.now()
        ))

if __name__ == "__main__":
    pytest.main([__file__])
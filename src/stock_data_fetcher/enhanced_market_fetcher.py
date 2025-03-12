import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import (
    ProviderConfig,
    RateLimitConfig,
    CacheConfig,
    MarketData
)
from market_data.exceptions import (
    MarketDataError,
    RateLimitError,
    CircuitBreakerError,
    ValidationError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegistry:
    """Registry for managing different market exchanges."""
    
    def __init__(self, config_dir: str = "data/config"):
        self.config_dir = Path(config_dir)
        self.markets: Dict[str, Set[str]] = {}
        self._load_markets()
    
    def _load_markets(self) -> None:
        """Load all available market listings from config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
            
        for file_path in self.config_dir.glob("*-listed.json"):
            market_name = file_path.stem.replace("-listed", "")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Handle both 'Symbol' and 'symbol' cases for compatibility
                    symbols = set()
                    for item in data:
                        if 'Symbol' in item:
                            symbols.add(item['Symbol'])
                        elif 'symbol' in item:
                            symbols.add(item['symbol'])
                        else:
                            logger.warning(f"Missing symbol key in item: {item}")
                    
                    self.markets[market_name] = symbols
                logger.info(f"Loaded {len(self.markets[market_name])} symbols for {market_name}")
            except Exception as e:
                logger.error(f"Error loading market {market_name}: {str(e)}")
    
    def get_symbols(self, market: str) -> Set[str]:
        """Get all symbols for a specific market."""
        return self.markets.get(market, set())
    
    def get_markets(self) -> List[str]:
        """Get list of all available markets."""
        return list(self.markets.keys())
    
    def validate_symbol(self, market: str, symbol: str) -> bool:
        """Validate if a symbol exists in the specified market."""
        return symbol in self.markets.get(market, set())

class EnhancedMarketDataFetcher:
    """Enhanced market data fetcher with support for multiple exchanges and backfilling."""
    
    def __init__(self, data_dir: str = "data", config_dir: str = "data/config"):
        self.data_dir = Path(data_dir)
        self.registry = MarketRegistry(config_dir)
        self.provider = self._setup_provider()
        
    def _setup_provider(self) -> YahooFinanceProvider:
        """Initialize the data provider with optimized configuration."""
        config = ProviderConfig(
            name="yahoo",
            rate_limit=RateLimitConfig(
                requests_per_second=2.0,
                burst_size=5,
                retry_after=60
            ),
            cache=CacheConfig(
                enabled=True,
                ttl=86400,
                directory=".cache"
            )
        )
        return YahooFinanceProvider(config)
    
    async def _get_existing_dates(self, market: str, symbol: str) -> Set[datetime]:
        """Get set of dates for which we already have data."""
        symbol_dir = self.data_dir / market / symbol
        if not symbol_dir.exists():
            return set()
            
        dates = set()
        for file_path in symbol_dir.glob("*.json"):
            try:
                date_str = file_path.stem
                dates.add(datetime.strptime(date_str, "%Y-%m-%d"))
            except ValueError:
                logger.warning(f"Invalid date format in filename: {file_path}")
        return dates
    
    async def _save_market_data(self, market: str, symbol: str, date: datetime, data: MarketData):
        """Save market data to the appropriate directory structure."""
        symbol_dir = self.data_dir / market / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            "open": float(data.open),
            "high": float(data.high),
            "low": float(data.low),
            "close": float(data.close),
            "volume": float(data.volume),
            "dividends": float(getattr(data, "dividends", 0) or 0),
            "stock_splits": float(getattr(data, "stock_splits", 0) or 0)
        }
        
        file_path = symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
            
    async def backfill_data(self, market: str, symbol: str, start_date: datetime, end_date: Optional[datetime] = None):
        """Backfill missing data for a symbol between specified dates."""
        if not self.registry.validate_symbol(market, symbol):
            raise ValueError(f"Invalid symbol {symbol} for market {market}")
            
        end_date = end_date or datetime.now()
        existing_dates = await self._get_existing_dates(market, symbol)
        
        # Create a set of all dates in the range
        current_date = start_date
        required_dates = set()
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only include weekdays
                required_dates.add(current_date)
            current_date += timedelta(days=1)
        
        # Find missing dates
        missing_dates = required_dates - existing_dates
        if not missing_dates:
            logger.info(f"No missing data for {symbol} between {start_date} and {end_date}")
            return
            
        logger.info(f"Fetching {len(missing_dates)} missing data points for {symbol}")
        
        # Fetch missing data in chunks to respect rate limits
        chunks = sorted(list(missing_dates))
        chunk_size = 30  # Fetch 30 days at a time
        
        for i in range(0, len(chunks), chunk_size):
            chunk = chunks[i:i + chunk_size]
            chunk_start = chunk[0]
            chunk_end = chunk[-1]
            
            try:
                result = await self.provider.get_history(
                    symbol,
                    start_date=chunk_start,
                    end_date=chunk_end
                )
                
                if result and result.data:
                    for data_point in result.data:
                        await self._save_market_data(
                            market,
                            symbol,
                            data_point.date,
                            data_point
                        )
                    logger.info(f"Saved chunk {i//chunk_size + 1} for {symbol}")
                
                await asyncio.sleep(1)  # Rate limiting
                
            except (RateLimitError, CircuitBreakerError) as e:
                logger.error(f"Rate limit error for {symbol}: {str(e)}")
                await asyncio.sleep(60)
                continue
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
    
    async def fetch_market_data(self, market: str, start_date: datetime, end_date: Optional[datetime] = None):
        """Fetch data for all symbols in a market with automatic backfilling."""
        if market not in self.registry.get_markets():
            raise ValueError(f"Invalid market: {market}")
            
        symbols = self.registry.get_symbols(market)
        logger.info(f"Fetching data for {len(symbols)} symbols from {market}")
        
        for symbol in symbols:
            try:
                await self.backfill_data(market, symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
            
            await asyncio.sleep(1)  # Rate limiting between symbols

async def main():
    fetcher = EnhancedMarketDataFetcher()
    
    # Example usage
    start_date = datetime(2024, 1, 1)
    market = "nasdaq"  # or "nyse", etc.
    
    # Fetch all symbols for a market
    await fetcher.fetch_market_data(market, start_date)
    
    # Or backfill specific symbol
    await fetcher.backfill_data(market, "AAPL", start_date)

if __name__ == "__main__":
    asyncio.run(main())
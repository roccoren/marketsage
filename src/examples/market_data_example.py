"""Example usage of the market data system."""
import asyncio
import logging
from datetime import datetime, timedelta

from market_data.models.common import (
    MarketSource,
    ProviderConfig,
    CacheConfig,
    RateLimitConfig
)
from market_data.providers.yahoo import YahooFinanceProvider
from market_data.loaders import JsonMarketSymbolLoader
from market_data.cache.disk import DiskCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run market data examples."""
    try:
        # Initialize components
        provider_config = ProviderConfig(
            name="yahoo",
            cache=CacheConfig(
                enabled=True,
                directory="./data/cache"
            ),
            rate_limit=RateLimitConfig(
                requests_per_second=0.1,  # One request every 10 seconds
                burst_size=1  # No concurrent requests
            )
        )
        
        cache = DiskCache(provider_config.cache)
        provider = YahooFinanceProvider(provider_config)
        symbol_loader = JsonMarketSymbolLoader()
        
        # Load market symbols
        nasdaq_symbols = await symbol_loader.load_symbols(MarketSource.NASDAQ)
        nyse_symbols = await symbol_loader.load_symbols(MarketSource.NYSE)
        
        logger.info(f"Loaded {len(nasdaq_symbols)} NASDAQ and {len(nyse_symbols)} NYSE symbols")
        
        # Get market metadata
        nasdaq_meta = await symbol_loader.get_market_metadata(MarketSource.NASDAQ)
        logger.info(f"NASDAQ Market Info: {nasdaq_meta}")
        
        # Example 1: Fetch historical data for a single symbol
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Add initial delay to ensure we're starting fresh with rate limits
        logger.info("Waiting 10 seconds before starting requests...")
        await asyncio.sleep(10)
        
        history = await provider.get_history(symbol, start_date, end_date)
        logger.info(f"\nHistorical data for {symbol}:")
        for point in history["data"][-5:]:  # Show last 5 days
            logger.info(f"Date: {point.date.date()} Close: ${point.close:.2f}")
            
        # Add delay between operations
        await asyncio.sleep(10)
            
        # Example 2: Fetch real-time quotes for multiple symbols
        symbols = ["MSFT", "GOOGL", "AAPL"]
        quotes = await provider.get_real_time_quotes_batch(symbols)
        
        logger.info("\nReal-time quotes:")
        for symbol, quote in quotes.items():
            logger.info(
                f"{symbol}: ${quote.price:.2f} "
                f"({quote.change_percent:+.2f}%) "
                f"Vol: {quote.volume:,}"
            )
            
        # Add delay between operations    
        await asyncio.sleep(10)
            
        # Example 3: Get company information
        ticker_info = await provider.get_ticker_info("NVDA")
        logger.info(f"\nCompany Info for NVDA:")
        logger.info(f"Name: {ticker_info['name']}")
        logger.info(f"Sector: {ticker_info['sector']}")
        logger.info(f"Industry: {ticker_info['industry']}")
        
        # Example 4: Handle errors gracefully
        try:
            await provider.get_real_time_quote("INVALID")
        except Exception as e:
            logger.error(f"Expected error for invalid symbol: {e}")
            
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
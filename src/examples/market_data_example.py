"""Example usage of the consolidated market data fetcher."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from market_data.core.market_fetcher import ConsolidatedMarketFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demonstrate_market_fetching():
    """Demonstrate various market data fetching capabilities."""
    fetcher = ConsolidatedMarketFetcher(
        data_dir="data",
        config_dir="config/market",
        cache_ttl_days=1,
        max_concurrent=3,
        batch_size=10,
        rate_limit=0.5  # 2 requests per second max
    )
    
    # List available markets
    markets = fetcher.get_available_markets()
    logger.info(f"Available markets: {', '.join(markets)}")
    
    # Fetch last week's data for AAPL
    start_date = datetime.now() - timedelta(days=7)
    market = "nasdaq"  # Example market
    symbol = "AAPL"
    
    logger.info(f"\nFetching last week's data for {symbol}...")
    date = datetime.now() - timedelta(days=1)  # Get yesterday's data
    
    data = await fetcher.get_market_data(market, symbol, date)
    if data:
        logger.info(f"\nData for {symbol} on {date.strftime('%Y-%m-%d')}:")
        logger.info(f"  Open: {data['open']:.2f}")
        logger.info(f"  High: {data['high']:.2f}")
        logger.info(f"  Low: {data['low']:.2f}")
        logger.info(f"  Close: {data['close']:.2f}")
        logger.info(f"  Volume: {data['volume']:,.0f}")
        
    # Demonstrate batch fetching for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    logger.info(f"\nFetching data for multiple symbols: {', '.join(symbols)}")
    
    await fetcher.fetch_market_data(
        market="nasdaq",
        start_date=start_date,
        symbols=symbols
    )
    
    logger.info("Data fetching demonstration completed.")

async def main():
    """Run the market data fetching demonstration."""
    try:
        await demonstrate_market_fetching()
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
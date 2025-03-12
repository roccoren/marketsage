import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from enhanced_market_fetcher import EnhancedMarketDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_basic_fetching():
    """Demonstrate basic market data fetching."""
    fetcher = EnhancedMarketDataFetcher()
    
    # Fetch recent data for NASDAQ
    start_date = datetime.now() - timedelta(days=30)  # Last 30 days
    try:
        logger.info("Fetching NASDAQ market data for the last 30 days")
        await fetcher.fetch_market_data("nasdaq", start_date)
    except Exception as e:
        logger.error(f"Error fetching NASDAQ data: {str(e)}")

async def demo_backfilling():
    """Demonstrate backfilling capabilities."""
    fetcher = EnhancedMarketDataFetcher()
    
    # Backfill specific symbols for a longer period
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2024, 1, 1)
    
    for symbol in symbols:
        try:
            logger.info(f"Backfilling data for {symbol} from {start_date}")
            await fetcher.backfill_data(
                "nasdaq",
                symbol,
                start_date
            )
        except Exception as e:
            logger.error(f"Error backfilling {symbol}: {str(e)}")
        
        await asyncio.sleep(1)  # Rate limiting between symbols

async def demo_multi_market():
    """Demonstrate handling multiple markets."""
    fetcher = EnhancedMarketDataFetcher()
    
    # Get list of available markets
    markets = fetcher.registry.get_markets()
    logger.info(f"Available markets: {markets}")
    
    # Fetch recent data for all markets
    start_date = datetime.now() - timedelta(days=7)  # Last week
    
    for market in markets:
        try:
            logger.info(f"Processing market: {market}")
            
            # Get number of symbols for this market
            symbols = fetcher.registry.get_symbols(market)
            logger.info(f"Found {len(symbols)} symbols in {market}")
            
            # Fetch market data
            await fetcher.fetch_market_data(market, start_date)
            
        except Exception as e:
            logger.error(f"Error processing {market}: {str(e)}")
        
        await asyncio.sleep(2)  # Rate limiting between markets

async def demo_gap_filling():
    """Demonstrate filling gaps in historical data."""
    fetcher = EnhancedMarketDataFetcher()
    
    # Example: Fill gaps for specific symbols
    gaps = [
        # (market, symbol, start_date, end_date)
        ("nasdaq", "AAPL", datetime(2024, 1, 1), datetime(2024, 2, 1)),
        ("nyse", "IBM", datetime(2024, 1, 1), datetime(2024, 2, 1))
    ]
    
    for market, symbol, start_date, end_date in gaps:
        try:
            logger.info(f"Filling gaps for {symbol} from {start_date} to {end_date}")
            await fetcher.backfill_data(
                market,
                symbol,
                start_date,
                end_date
            )
        except Exception as e:
            logger.error(f"Error filling gaps for {symbol}: {str(e)}")
        
        await asyncio.sleep(1)

def verify_data_structure():
    """Verify the data directory structure and list available data."""
    data_dir = Path("data")
    
    print("\nData Directory Structure:")
    print("========================")
    
    if not data_dir.exists():
        print("Data directory not found!")
        return
        
    for market_dir in data_dir.iterdir():
        if market_dir.is_dir():
            print(f"\nMarket: {market_dir.name}")
            symbol_count = sum(1 for _ in market_dir.iterdir() if _.is_dir())
            print(f"Number of symbols: {symbol_count}")
            
            # Show sample of symbols
            sample_symbols = list(market_dir.iterdir())[:5]
            if sample_symbols:
                print("Sample symbols:")
                for symbol_dir in sample_symbols:
                    if symbol_dir.is_dir():
                        data_files = list(symbol_dir.glob("*.json"))
                        print(f"  - {symbol_dir.name}: {len(data_files)} data points")

async def main():
    """Run all demos."""
    print("Enhanced Market Data Fetcher Demo")
    print("================================")
    
    # Basic fetching
    print("\n1. Basic Market Data Fetching")
    await demo_basic_fetching()
    
    # Backfilling
    print("\n2. Backfilling Historical Data")
    await demo_backfilling()
    
    # Multiple markets
    print("\n3. Multi-Market Processing")
    await demo_multi_market()
    
    # Gap filling
    print("\n4. Filling Data Gaps")
    await demo_gap_filling()
    
    # Verify results
    print("\n5. Verifying Data Structure")
    verify_data_structure()

if __name__ == "__main__":
    asyncio.run(main())
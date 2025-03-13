"""Example demonstrating enhanced market data fetching capabilities."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from market_data.providers.enhanced_fetcher import EnhancedMarketFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def fetch_multiple_markets():
    """Demonstrate fetching data from multiple markets with different configurations."""
    # Initialize the enhanced fetcher with custom settings
    fetcher = EnhancedMarketFetcher(
        data_dir="data",
        config_dir="config/market",
        max_concurrent=5,
        batch_size=20,
        requests_per_minute=60,
        retry_attempts=3
    )

    # Set up dates for data fetching
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Market configurations for examples
    market_configs = {
        "nyse": ["JPM", "BAC", "WMT", "PFE", "KO"],
        "nasdaq": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "hkex": ["0001.HK", "0002.HK", "0003.HK", "0005.HK", "0011.HK"],
        "asx": ["BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX"]
    }

    # Example 1: Fetch data from all markets
    for market, symbols in market_configs.items():
        logger.info(f"\nFetching {market.upper()} data...")
        try:
            await fetcher.fetch_market_data(
                market=market,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
        except Exception as e:
            logger.error(f"Error fetching {market} data: {str(e)}")

    # Example 2: Demonstrate parallel fetching with custom chunk sizes
    logger.info("\nDemonstrating parallel fetching with custom chunk sizes...")
    tasks = []
    for market, symbols in market_configs.items():
        # Use different chunk sizes for different markets
        chunk_size = 2 if market in ["hkex", "asx"] else 3
        tasks.append(
            fetcher.fetch_market_data(
                market=market,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols[:2],  # Using fewer symbols for demonstration
                chunk_size=chunk_size
            )
        )
    await asyncio.gather(*tasks)

    # Example 3: Retrieve and compare data across markets
    logger.info("\nRetrieving specific market data examples...")
    sample_symbols = {
        "nyse": "JPM",
        "nasdaq": "AAPL",
        "hkex": "0001.HK",
        "asx": "BHP.AX"
    }

    for market, symbol in sample_symbols.items():
        data = await fetcher.get_market_data(
            market=market,
            symbol=symbol,
            date=end_date
        )
        if data:
            logger.info(f"{market.upper()} - {symbol} data for {end_date.date()}: {data}")

def check_data_structure():
    """Verify the organized data storage structure."""
    data_dir = Path("data")
    
    # Check data organization for all markets
    markets = ["nyse", "nasdaq", "hkex", "asx"]
    for market in markets:
        market_dir = data_dir / market
        if market_dir.exists():
            logger.info(f"\nData structure for {market.upper()}:")
            symbol_dirs = list(market_dir.glob("*"))
            if symbol_dirs:
                for symbol_dir in symbol_dirs:
                    if symbol_dir.is_dir():
                        file_count = len(list(symbol_dir.glob("*.json")))
                        logger.info(f"- {symbol_dir.name}: {file_count} data files")
            else:
                logger.info("No data files found")

async def main():
    """Run the enhanced market data fetching examples."""
    try:
        logger.info("Starting enhanced market data fetching examples...")
        
        # Create data directory structure
        data_dir = Path("data")
        for market in ["nyse", "nasdaq", "hkex", "asx"]:
            (data_dir / market).mkdir(parents=True, exist_ok=True)
        
        # Run examples
        await fetch_multiple_markets()
        check_data_structure()
        
        logger.info("\nExamples completed successfully")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
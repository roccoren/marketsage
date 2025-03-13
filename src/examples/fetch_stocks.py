import asyncio
import logging
from datetime import datetime, timedelta
from stock_data_fetcher.optimized_market_fetcher import OptimizedMarketFetcher

# Configure logging to see the progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_stocks():
    # Initialize the fetcher with conservative settings to avoid rate limits
    fetcher = OptimizedMarketFetcher(
        max_concurrent=2,  # Maximum concurrent requests
        batch_size=5,     # Process 5 symbols at a time
        rate_limit_delay=3.0  # 3 seconds between requests
    )

    # Define time range - last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # List of stock symbols to fetch
    symbols = ["AAPL", "MSFT", "GOOGL"]  # Example tech stocks
    
    try:
        logger.info(f"Fetching market data for: {', '.join(symbols)}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Wait a moment before starting (good practice)
        logger.info("Initializing...")
        await asyncio.sleep(2)
        
        # Fetch data from NASDAQ
        await fetcher.fetch_market_data(
            market="nasdaq",
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        logger.info("Data fetch completed successfully!")
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(fetch_stocks())
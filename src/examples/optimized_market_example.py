import asyncio
import logging
from datetime import datetime, timedelta
from stock_data_fetcher.optimized_market_fetcher import OptimizedMarketFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize fetcher with conservative settings
        fetcher = OptimizedMarketFetcher(
            max_concurrent=2,     # Limit concurrent downloads
            batch_size=5,        # Process 5 symbols at a time
            rate_limit_delay=3.0  # 3 seconds between requests
        )
        
        # Get available markets
        markets = fetcher.get_available_markets()
        logger.info(f"Available markets: {markets}")
        
        # Load some test symbols
        test_symbols = {
            "NASDAQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "NYSE": ["BAC", "JPM", "WFC", "PFE", "KO"]
        }
        
        symbols_info = []
        for market, symbols in test_symbols.items():
            available = fetcher.get_market_symbols(market.lower())
            valid_symbols = [s for s in symbols if s in available]
            symbols_info.append(f"{len(valid_symbols)} {market}")
        
        logger.info(f"Loaded {' and '.join(symbols_info)} symbols")
        
        # Set date range for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for market, symbols in test_symbols.items():
            try:
                logger.info(f"\nProcessing {market} Market Data")
                logger.info(f"Symbols: {', '.join(symbols)}")
                
                # Initial delay to respect rate limits
                logger.info("Waiting 10 seconds before starting requests...")
                await asyncio.sleep(10)
                
                # Fetch data for test symbols
                await fetcher.fetch_market_data(
                    market=market.lower(),
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols
                )
                
                logger.info(f"Completed {market} data fetch")
                
            except Exception as e:
                logger.error(f"Error processing {market} market: {str(e)}")
                continue
            
            # Add delay between markets
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
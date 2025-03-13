"""Standalone market data fetcher example."""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketFetcher:
    """Simplified market data fetcher."""
    
    def __init__(
        self,
        data_dir: str = "data",
        max_concurrent: int = 3
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict]:
        """Fetch data for a single symbol."""
        async with self.semaphore:
            try:
                logger.info(f"Downloading data for {symbol}")
                loop = asyncio.get_running_loop()
                
                def download():
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.info
                        hist = ticker.history(
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            interval="1d"
                        )
                        return hist if not hist.empty else None
                    except Exception as e:
                        logger.error(f"Error downloading {symbol}: {str(e)}")
                        return None
                
                df = await loop.run_in_executor(self.thread_pool, download)
                await asyncio.sleep(1.0)  # Basic rate limiting
                return df
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                return None

    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[dict]]:
        """
        Fetch market data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date for data fetching
            end_date: Optional end date (defaults to current date)
        """
        if not symbols:
            raise ValueError("No symbols provided")
            
        end_date = end_date or datetime.now()
        logger.info(f"Fetching data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            df = await self._fetch_symbol_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                symbol_results = []
                
                for date, row in df.iterrows():
                    # Skip if any essential data is missing
                    if any(pd.isna(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                        continue
                        
                    data = {
                        "date": date.strftime('%Y-%m-%d'),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume'])
                    }
                    symbol_results.append(data)
                
                if symbol_results:
                    results[symbol] = symbol_results
                    logger.info(f"Successfully processed {symbol}")
        
        return results

async def demonstrate_fetch():
    """Demonstrate market data fetching."""
    fetcher = MarketFetcher(
        data_dir="./data",
        max_concurrent=2
    )
    
    # Test with a single well-known symbol first
    symbols = ["AAPL"]
    start_date = datetime.now() - timedelta(days=7)
    logger.info(f"Fetching last week's data for {symbols}...")
    
    results = await fetcher.fetch_market_data(
        symbols,
        start_date
    )
    
    # Print results
    for symbol, data in results.items():
        if data:
            latest = data[-1]  # Get most recent data point
            logger.info(f"\nLatest data for {symbol} ({latest['date']}):")
            logger.info(f"  Open: {latest['open']:.2f}")
            logger.info(f"  Close: {latest['close']:.2f}")
            logger.info(f"  Volume: {latest['volume']:,.0f}")

async def main():
    """Run the example demonstrations."""
    logger.info("Starting market data fetcher example...")
    
    try:
        # Create data directory if it doesn't exist
        Path("./data").mkdir(exist_ok=True)
        
        # Run demonstration
        await demonstrate_fetch()
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        raise
    
    logger.info("Example completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
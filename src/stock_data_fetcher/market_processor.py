import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from enhanced_market_fetcher import EnhancedMarketDataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketProcessor:
    """Process market data based on config files."""
    
    def __init__(self):
        self.fetcher = EnhancedMarketDataFetcher(cache_dir="data")
        self.config_dir = Path("data/config")
        self.data_dir = Path("data")
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        markets = ["nyse", "nasdaq"]
        for market in markets:
            (self.data_dir / market).mkdir(parents=True, exist_ok=True)
    
    def _load_market_config(self, market: str) -> List[Dict]:
        """Load market configuration from JSON file."""
        config_file = self.config_dir / f"{market}-listed.json"
        try:
            with open(config_file, 'r') as f:
                # Load full config data
                config_data = json.load(f)
                
                # Transform into simple symbol list format
                symbols_list = [{"symbol": item["symbol"]} for item in config_data]
                
                logger.debug(f"Loaded {len(symbols_list)} symbols for {market}")
                return symbols_list
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_file}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_file}")
            return []
        except KeyError:
            logger.error(f"Invalid config format in {config_file}")
            return []
    
    async def process_market(self, market: str):
        """Process data for a specific market."""
        logger.info(f"Processing {market.upper()} market data")
        
        # Load and transform market configuration
        symbols = self._load_market_config(market)
        if not symbols:
            logger.error(f"No valid symbols found for {market}")
            return
        
        # Create temporary symbols file
        temp_file = self.data_dir / f"{market}_symbols.json"
        try:
            # Write transformed symbols to temporary file
            with open(temp_file, 'w') as f:
                json.dump(symbols, f)
            
            # Get start date (30 days ago)
            start_date = datetime.now() - timedelta(days=30)
            
            # Fetch market data using temporary symbols file
            await self.fetcher.fetch_market_data(
                market=market,
                symbols_file=str(temp_file),
                start_date=start_date
            )
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
    
    async def process_all_markets(self):
        """Process data for all configured markets."""
        self._ensure_directories()
        
        # Process both markets concurrently
        tasks = [
            self.process_market("nyse"),
            self.process_market("nasdaq")
        ]
        
        await asyncio.gather(*tasks)

async def main():
    processor = MarketProcessor()
    await processor.process_all_markets()

if __name__ == "__main__":
    asyncio.run(main())
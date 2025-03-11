"""
Test script to demonstrate the MarketDataFetcher with example symbols.
"""
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from market_data_fetcher import MarketDataFetcher

async def test_fetch():
    # Create separate test data directory
    test_data_dir = "test_data"
    
    # Load example symbols
    with open("example-symbols.json", "r") as f:
        symbols = json.load(f)
    
    # Create temporary symbol files for each market
    for market in ["nyse", "nasdaq"]:
        with open(f"test_{market}_symbols.json", "w") as f:
            json.dump(symbols[market], f, indent=2)
    
    # Initialize fetcher with test directory
    fetcher = MarketDataFetcher(cache_dir=test_data_dir)
    
    # Fetch last 7 days of data
    start_date = datetime.now() - timedelta(days=7)
    
    try:
        # Fetch data for both markets
        for market in ["nyse", "nasdaq"]:
            print(f"\nFetching {market.upper()} data...")
            await fetcher.fetch_market_data(
                market,
                f"test_{market}_symbols.json",
                start_date
            )
        
        # Verify the data structure
        test_dir = Path(test_data_dir)
        if test_dir.exists():
            print("\nGenerated data structure:")
            for market in test_dir.iterdir():
                print(f"\n{market.name}/")
                for symbol in market.iterdir():
                    print(f"  {symbol.name}/")
                    for date_file in symbol.iterdir():
                        print(f"    {date_file.name}")
    
    finally:
        # Cleanup temporary symbol files
        for market in ["nyse", "nasdaq"]:
            temp_file = Path(f"test_{market}_symbols.json")
            if temp_file.exists():
                temp_file.unlink()

if __name__ == "__main__":
    print("Starting MarketDataFetcher test...")
    print("This will download 7 days of data for example stocks")
    print("Data will be saved to ./test_data directory")
    
    asyncio.run(test_fetch())
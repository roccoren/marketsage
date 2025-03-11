"""Example demonstrating usage of the market data package with Yahoo Finance."""
import os
from datetime import datetime, timedelta
import asyncio
from market_data.providers.yahoo import YahooFinanceProvider
from market_data.models.common import (
    ProviderConfig,
    CacheConfig,
    RateLimitConfig,
    HistoricalData
)
from market_data.exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError
)

async def main():
    # Ensure cache directory exists
    cache_dir = "./data/cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize provider configuration
    config = ProviderConfig(
        name="yahoo",
        cache=CacheConfig(
            enabled=True,
            directory=cache_dir,
            ttl=24 * 3600  # 1 day cache
        ),
        rate_limit=RateLimitConfig(
            requests_per_second=0.2,  # Very conservative rate limiting
            burst_size=1,
            retry_after=5  # Wait 5 seconds after hitting rate limit
        )
    )
    
    # Initialize the provider
    provider = YahooFinanceProvider(config)

    # Define time range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last week of data
    
    # Example symbols with exchange information
    symbols = [
        "AAPL",  # Apple Inc.
        "MSFT",  # Microsoft Corporation
        "GOOGL"  # Alphabet Inc.
    ]
    
    try:
        print("Fetching historical data...\n")
        
        for symbol in symbols:
            try:
                print(f"Processing {symbol}...")
                
                # First try to get ticker info to validate symbol
                try:
                    info = await provider.get_ticker_info(symbol)
                    print(f"Found ticker: {info.name} ({info.exchange})")
                except Exception as e:
                    print(f"Could not verify ticker {symbol}: {str(e)}")
                    continue

                # Then fetch historical data
                print(f"Fetching historical data for {symbol}...")
                result = await provider.get_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not result or not isinstance(result, HistoricalData) or not result.data:
                    print(f"No data available for {symbol}\n")
                    continue
                
                # Display summary
                print(f"\nData summary for {symbol}:")
                print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
                print(f"Number of data points: {len(result.data)}")
                
                # Show latest trading day
                latest = result.data[-1]
                print(f"\nLatest trading day ({latest.date.strftime('%Y-%m-%d')}):")
                print(f"Open:     ${latest.open:.2f}")
                print(f"High:     ${latest.high:.2f}")
                print(f"Low:      ${latest.low:.2f}")
                print(f"Close:    ${latest.close:.2f}")
                print(f"Volume:   {latest.volume:,}")
                if latest.adjusted_close:
                    print(f"Adj Close: ${latest.adjusted_close:.2f}")
                print("\n" + "="*50 + "\n")
                
                # Add delay between requests to respect rate limits
                await asyncio.sleep(5)
                
            except ValidationError as e:
                print(f"Invalid symbol {symbol}: {str(e)}\n")
            except DataNotFoundError as e:
                print(f"No data found for {symbol}: {str(e)}\n")
            except ConnectionError as e:
                print(f"Connection error for {symbol}: {str(e)}\n")
            except Exception as e:
                print(f"Unexpected error processing {symbol}: {str(e)}\n")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        # Cleanup resources
        if hasattr(provider, '__del__'):
            provider.__del__()

if __name__ == "__main__":
    print("Market Data Example")
    print("==================")
    print("This example demonstrates fetching historical market data")
    print("Make sure you have installed required packages:")
    print("pip install -r config/requirements.txt")
    print("==================\n")
    
    # Run the async main function
    asyncio.run(main())
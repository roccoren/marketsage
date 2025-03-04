"""
Stock Data Downloader Example

This script demonstrates how to use the StockDataDownloader class to download 
daily stock data from various markets (NYSE, NASDAQ, ASX, HKEX) and save it 
in JSON format.
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.market_analysis.data_downloader import StockDataDownloader


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download stock data from various markets')
    
    parser.add_argument('--ticker', '-t', type=str, help='Ticker symbol to download')
    parser.add_argument('--market', '-m', choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX'], 
                        help='Market to download from')
    parser.add_argument('--start', '-s', type=str, default=None, 
                        help='Start date in YYYY-MM-DD format (default: 30 days ago)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output directory for downloaded data')
    parser.add_argument('--multiple', '-mt', type=str, default=None, 
                        help='Comma-separated list of ticker symbols to download')
    parser.add_argument('--markets', '-ms', type=str, default=None,
                        help='Comma-separated list of markets to download (NYSE,NASDAQ,ASX,HKEX)')
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                        help='Use parallel downloading (default: True)')
    parser.add_argument('--no-parallel', '-np', action='store_false', dest='parallel',
                        help='Disable parallel downloading')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                        help='Number of stocks to download in each batch (default: 10)')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                        help='Maximum number of worker threads for parallel downloading')
    parser.add_argument('--no-daily', '-nd', action='store_false', dest='save_daily',
                        help='Do not save individual daily files (only save consolidated files)')
    parser.add_argument('--benchmark', '-bm', action='store_true',
                        help='Run performance benchmark for different configurations')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = StockDataDownloader(max_workers=args.max_workers)
    
    # Set default start date to 30 days ago if not specified
    if args.start is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        start_date = args.start
    
    # Benchmarking mode
    if args.benchmark:
        run_benchmark(downloader, start_date)
        return
    
    # Check for multiple markets download
    if args.markets:
        markets = [m.strip() for m in args.markets.split(',')]
        print(f"Downloading data for markets: {', '.join(markets)} starting from {start_date}")
        print(f"Using parallel: {args.parallel}, batch size: {args.batch_size}, saving daily files: {args.save_daily}")
        
        start_time = time.time()
        results = downloader.batch_download_markets(
            markets,
            start_date,
            parallel=args.parallel,
            batch_size=args.batch_size,
            save_daily=args.save_daily
        )
        elapsed = time.time() - start_time
        
        print(f"Downloaded data for {sum(len(market_data) for market_data in results.values())} stocks across {len(results)} markets")
        print(f"Total time: {elapsed:.2f} seconds")
        return
        
    # Single ticker download
    if args.ticker and args.market:
        # Download a single stock
        print(f"Downloading data for {args.ticker} from {args.market} starting from {start_date}")
        
        start_time = time.time()
        result = downloader.download_stock_data(
            args.ticker, 
            args.market, 
            start_date,
            save_dir=args.output,
            save_daily=args.save_daily
        )
        elapsed = time.time() - start_time
        
        print(f"Successfully downloaded data for {args.ticker}")
        print(f"Data contains {len(result['history'])} days of history")
        print(f"Time taken: {elapsed:.2f} seconds")
        
    # Multiple tickers download
    elif args.multiple and args.market:
        # Download multiple stocks
        tickers = [t.strip() for t in args.multiple.split(',')]
        print(f"Downloading data for {len(tickers)} stocks from {args.market} starting from {start_date}")
        print(f"Using parallel: {args.parallel}, batch size: {args.batch_size}, saving daily files: {args.save_daily}")
        
        start_time = time.time()
        results = downloader.download_multiple_stocks(
            tickers, 
            args.market, 
            start_date,
            parallel=args.parallel,
            batch_size=args.batch_size,
            save_daily=args.save_daily
        )
        elapsed = time.time() - start_time
        
        print(f"Successfully downloaded data for {len(results)} stocks")
        print(f"Total time: {elapsed:.2f} seconds")
        
    # Download all stocks in a market from a listing file
    elif args.market:
        try:
            print(f"Downloading stock data for {args.market} market stocks starting from {start_date}")
            print(f"Using parallel: {args.parallel}, batch size: {args.batch_size}, saving daily files: {args.save_daily}")
            
            start_time = time.time()
            results = downloader.download_market_list(
                args.market,
                start_date=start_date,
                parallel=args.parallel,
                batch_size=args.batch_size,
                save_daily=args.save_daily
            )
            elapsed = time.time() - start_time
            
            print(f"Successfully downloaded data for {len(results)} stocks")
            print(f"Total time: {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Make sure you have the market listing file available in the data directory.")
    
    else:
        print("Please specify a ticker symbol and market or use --markets option. Use --help for more information.")
        
        # Show an example
        print("\nExample usage:")
        print("  Download a single stock:")
        print("    python stock_data_download_example.py -t AAPL -m NASDAQ -s 2023-01-01")
        print("\n  Download multiple stocks:")
        print("    python stock_data_download_example.py -m NASDAQ -mt AAPL,MSFT,GOOGL -s 2023-01-01 -p -b 5")
        print("\n  Download multiple markets:")
        print("    python stock_data_download_example.py -ms NASDAQ,NYSE -s 2023-01-01 -p -b 10")
        print("\n  Run benchmark:")
        print("    python stock_data_download_example.py -bm")


def run_benchmark(downloader: StockDataDownloader, start_date: str):
    """Run performance benchmark for different downloading configurations"""
    print("=== Running Performance Benchmark ===")
    
    # Define test parameters
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]
    market = "NASDAQ"
    
    # Test scenarios
    scenarios = [
        {"name": "Sequential", "parallel": False, "batch_size": 10, "save_daily": True},
        {"name": "Parallel (batch=5)", "parallel": True, "batch_size": 5, "save_daily": True},
        {"name": "Parallel (batch=10)", "parallel": True, "batch_size": 10, "save_daily": True},
        {"name": "Parallel (no daily files)", "parallel": True, "batch_size": 10, "save_daily": False},
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        print(f"Configuration: parallel={scenario['parallel']}, batch_size={scenario['batch_size']}, save_daily={scenario['save_daily']}")
        
        # Run test with current configuration
        start_time = time.time()
        data = downloader.download_multiple_stocks(
            tickers, 
            market, 
            start_date,
            parallel=scenario["parallel"],
            batch_size=scenario["batch_size"],
            save_daily=scenario["save_daily"]
        )
        elapsed = time.time() - start_time
        
        results.append({
            "scenario": scenario["name"],
            "time": elapsed,
            "stocks": len(data)
        })
        
        print(f"Completed in {elapsed:.2f} seconds, successfully downloaded {len(data)} stocks")
    
    # Print summary
    print("\n=== Benchmark Results ===")
    print(f"Downloaded {len(tickers)} stocks from {market} starting from {start_date}")
    print("Configuration | Time (seconds) | Stocks")
    print("-------------|----------------|-------")
    for result in sorted(results, key=lambda x: x["time"]):
        print(f"{result['scenario']:<13} | {result['time']:<14.2f} | {result['stocks']}")


if __name__ == "__main__":
    main()
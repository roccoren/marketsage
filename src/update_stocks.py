# Import yfinance config first to set urllib3 settings
from utils import yfinance_config

import csv
import json
import argparse
import asyncio
import aiohttp
import logging
import time
import os
import signal
from datetime import date, datetime, timedelta
import re
from utils.async_utils import YahooFinanceClient
from contextlib import asynccontextmanager

# Enhanced logging configuration
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
root_handler = root_logger.hasHandlers() and root_logger.handlers[0]
if root_handler:
    root_logger.removeHandler(root_handler)
root_logger.addHandler(console_handler)

# Add file handler
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/stock_update.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

# Set yfinance logger to INFO level
yfinance_logger = logging.getLogger('yfinance')
yfinance_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Global references
loop = None

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, calls_per_second=5):
        self.calls_per_second = calls_per_second
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to maintain rate limit."""
        async with self.lock:
            now = time.time()
            if self.last_call:
                time_since_last = now - self.last_call
                if time_since_last < self.minimum_interval:
                    await asyncio.sleep(self.minimum_interval - time_since_last)
            self.last_call = time.time()

def print_progress(current, total, prefix='Progress:', suffix='Complete'):
    """Print progress bar."""
    length = 50
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    percent = current / total * 100
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='\r')
    if current == total:
        print()

def print_summary(exchange: str, results: dict, start_time: float):
    """Print a formatted summary of exchange processing results."""
    elapsed = time.time() - start_time
    successful = len([v for v in results.values() if v])
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print("\n" + "="*80)
    print(f"Exchange Summary: {exchange}")
    print("="*80)
    print(f"Total symbols processed: {total}")
    print(f"Successfully updated:    {successful}")
    print(f"Failed:                 {total - successful}")
    print(f"Success rate:           {success_rate:.1f}%")
    print(f"Time elapsed:           {elapsed:.1f} seconds")
    print(f"Average time/symbol:    {(elapsed/total):.2f} seconds")
    print("="*80 + "\n")

    logger.info(f"\nExchange {exchange} completed:")
    logger.info(f"Success rate: {success_rate:.1f}% ({successful}/{total})")
    logger.info(f"Time elapsed: {elapsed:.1f}s")

async def get_market_caps(client, symbols, rate_limiter):
    """Get market caps for symbols with progress tracking."""
    results = {}
    total = len(symbols)
    
    print("\nFetching market caps...")
    for i, symbol in enumerate(symbols):
        try:
            await rate_limiter.acquire()
            info = await client.get_ticker_info(symbol)
            market_cap = info['price'].get('marketCap') if info else None
            results[symbol] = market_cap
            print_progress(i + 1, total, prefix='Market Caps:', suffix='Complete')
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            results[symbol] = None
    print()
    return results

async def process_exchange(exchange_name, symbols, start_date, incremental, data_dir, chunk_size=200):
    """Process all symbols for a single exchange."""
    if not symbols:
        logger.info(f"No symbols to process for {exchange_name}")
        return {}

    exchange_dir = os.path.join(data_dir, exchange_name)
    os.makedirs(exchange_dir, exist_ok=True)
    
    today = datetime.today()
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid start date format. Please use YYYY-MM-DD format")
            return {}
    else:
        start = today - timedelta(days=365)

    logger.info(f"\nProcessing {exchange_name}:")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Period:  {start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
    
    chunk_results = {}
    process_start_time = time.time()
    rate_limiter = RateLimiter(calls_per_second=5)
    client = None
    
    try:
        client = YahooFinanceClient(data_dir=exchange_dir)
        await client.__aenter__()
        total_processed = 0
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
            
            for attempt in range(3):  # Max 3 retries
                try:
                    await rate_limiter.acquire()
                    result = await client.get_history_batch(
                        chunk,
                        start_date=start.strftime('%Y-%m-%d') if not incremental else None,
                        end_date=today.strftime('%Y-%m-%d'),
                        incremental=incremental
                    )
                    if result:
                        chunk_results.update(result)
                        total_processed += len(result)
                        print_progress(total_processed, len(symbols), 
                                     prefix=f'{exchange_name} Progress:')
                        break
                    elif attempt == 2:  # Last attempt
                        logger.error(f"Failed to process chunk {chunk_num}/{total_chunks}")
                    else:
                        wait_time = (2 ** attempt) + 1
                        logger.info(f"Retrying chunk {chunk_num} in {wait_time}s")
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                    await asyncio.sleep((2 ** attempt) + 1)

        if not chunk_results:
            logger.error(f"No successful results for {exchange_name}")
            return {}

        # Get market caps with separate progress tracking
        active_statuses = await get_market_caps(
            client,
            [s for s, d in chunk_results.items() if d is not None],
            rate_limiter
        )

        print_summary(exchange_name, chunk_results, process_start_time)
        return active_statuses
        
    except Exception as e:
        logger.error(f"Error processing {exchange_name}: {e}")
        return {}
    finally:
        if client:
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
            await asyncio.sleep(1)  # Ensure cleanup completes

async def update_all_listings(exchanges=None, incremental=False, start_date=None, symbol=None):
    """Update stock listings for specified exchanges."""
    if symbol:
        logger.info(f"Processing single symbol: {symbol}")
        results = await process_exchange("UNKNOWN", [symbol], start_date, incremental, "data")
        return len(results) > 0

    all_exchanges = {
        'NYSE': 'data/nyse_filtered_listings.json',
        'NASDAQ': 'data/nasdaq_filtered_listings.json',
        'ASX': 'data/asx_filtered_listings.json'
    }
    
    exchanges_to_process = exchanges if exchanges else all_exchanges.keys()
    logger.info(f"Starting {'incremental' if incremental else 'full'} update for: "
               f"{', '.join(exchanges_to_process)}")
    
    results = {}
    success = True
    overall_start_time = time.time()
    
    try:
        for exchange in exchanges_to_process:
            try:
                filtered_file = all_exchanges[exchange]
                if not os.path.exists(filtered_file):
                    logger.error(f"Missing filtered listings file: {filtered_file}")
                    continue

                with open(filtered_file, 'r', encoding='utf-8') as f:
                    listings = json.load(f)
                    logger.info(f"Loaded {len(listings)} symbols from {filtered_file}")
                    
                symbols = [listing['symbol'] for listing in listings]
                active_statuses = await process_exchange(exchange, symbols, start_date, incremental, "data")
                
                if active_statuses:
                    updated_listings = []
                    for listing in listings:
                        if listing['symbol'] in active_statuses:
                            listing['market_cap'] = active_statuses[listing['symbol']]
                            updated_listings.append(listing)
                    
                    with open(filtered_file, 'w', encoding='utf-8') as f:
                        json.dump(updated_listings, f, indent=4)
                    
                    results[exchange] = True
                else:
                    results[exchange] = False
                    success = False
                    logger.error(f"No active statuses returned for {exchange}")
                
                # Ensure proper cleanup between exchanges
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error processing {exchange}: {str(e)}", exc_info=True)
                results[exchange] = False
                success = False
                
    except asyncio.CancelledError:
        logger.info("Update process cancelled")
        raise
    finally:
        total_time = time.time() - overall_start_time
        logger.info(f"\nProcess completed in {total_time:.1f} seconds")
        logger.info(f"Results: {results}")
    
    return success

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    global loop
    logger.info("Received interrupt signal, cleaning up...")
    if loop and loop.is_running():
        loop.stop()
    raise KeyboardInterrupt

def main(exchanges=None, incremental=False, start_date=None, symbol=None):
    """Main entry point with proper asyncio handling."""
    global loop
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            update_all_listings(exchanges, incremental, start_date, symbol)
        )
    except KeyboardInterrupt:
        logger.info("Update process stopped by user")
        return False
    except Exception as e:
        logger.error(f"Error during update: {e}")
        return False
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            if loop and loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # Required for macOS
        
        parser = argparse.ArgumentParser(
            description='Update stock listings from various exchanges',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument('--exchanges', nargs='+', choices=['NYSE', 'NASDAQ', 'ASX'],
                          help='Specify exchanges to update (e.g., NYSE NASDAQ)')
        parser.add_argument('--incremental', action='store_true', 
                          help='Only fetch data since the last available date')
        parser.add_argument('--start-date', type=str, 
                          help='Start date for data retrieval (format: YYYY-MM-DD)')
        parser.add_argument('--symbol', type=str,
                          help='Update a specific stock symbol')
        args = parser.parse_args()
        
        if args.start_date and args.incremental:
            logger.error("Cannot use both --start-date and --incremental options")
            exit(1)
            
        main(args.exchanges, args.incremental, args.start_date, args.symbol)
        
    except KeyboardInterrupt:
        logger.info("Process terminated by user")
    except Exception as e:
        logger.error(f"Process error: {e}")
        exit(1)
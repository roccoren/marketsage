# Import yfinance config first to set urllib3 settings
from . import yfinance_config as config

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
import pytz
import re
import random
from .async_utils import YahooFinanceClient, UpdateProgressTracker
from contextlib import asynccontextmanager

# Enhanced logging configuration
log_format = config.DEFAULT_CONFIG['log_format']
log_dir = config.DEFAULT_CONFIG['log_directory']

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, config.DEFAULT_CONFIG['log_level']))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
root_handler = root_logger.hasHandlers() and root_logger.handlers[0]
if root_handler:
    root_logger.removeHandler(root_handler)
root_logger.addHandler(console_handler)

# Add file handler
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'stock_update.log'), mode='w')
file_handler.setLevel(getattr(logging, config.DEFAULT_CONFIG['log_level']))
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

# Set yfinance logger to INFO level
yfinance_logger = logging.getLogger('yfinance')
yfinance_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Global references
loop = None
scheduler_task = None

class RateLimiter:
    """Advanced rate limiter for API calls with jitter."""
    def __init__(self, calls_per_second=None):
        rate_limit_ms = config.DEFAULT_CONFIG['rate_limit_ms']
        self.calls_per_second = calls_per_second or (1000 / rate_limit_ms)
        self.minimum_interval = 1.0 / self.calls_per_second
        self.last_call = 0
        self.lock = asyncio.Lock()
        # Track consecutive failures to dynamically adjust rate
        self.consecutive_failures = 0
        self.backoff_factor = 1.0

    async def acquire(self):
        """Wait if necessary to maintain rate limit with dynamic backoff."""
        async with self.lock:
            now = time.time()
            if self.last_call:
                time_since_last = now - self.last_call
                wait_time = self.minimum_interval * self.backoff_factor
                
                # Add jitter (±10%) to avoid request clustering
                jitter = wait_time * (random.random() * 0.2 - 0.1)
                wait_time += jitter
                
                if time_since_last < wait_time:
                    await asyncio.sleep(wait_time - time_since_last)
            self.last_call = time.time()
    
    def report_success(self):
        """Report a successful API call to potentially reduce backoff."""
        self.consecutive_failures = max(0, self.consecutive_failures - 1)
        if self.consecutive_failures == 0:
            self.backoff_factor = max(1.0, self.backoff_factor * 0.8)
    
    def report_failure(self):
        """Report a failed API call to increase backoff."""
        self.consecutive_failures += 1
        if self.consecutive_failures > 3:
            self.backoff_factor = min(10.0, self.backoff_factor * 1.5)

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
            rate_limiter.report_success()
            print_progress(i + 1, total, prefix='Market Caps:', suffix='Complete')
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            rate_limiter.report_failure()
            results[symbol] = None
    print()
    return results

async def process_exchange(exchange_name, symbols, start_date, incremental, data_dir, chunk_size=None):
    """Process all symbols for a single exchange."""
    if not symbols:
        logger.info(f"No symbols to process for {exchange_name}")
        return {}

    exchange_dir = os.path.join(data_dir, exchange_name)
    os.makedirs(exchange_dir, exist_ok=True)
    
    today = datetime.today()
    if start_date:
        try:
            start = datetime.strptime(start_date, config.DEFAULT_CONFIG['output_format']['date_format'])
        except ValueError:
            logger.error(f"Invalid start date format. Please use {config.DEFAULT_CONFIG['output_format']['date_format']} format")
            return {}
    else:
        start = today - timedelta(days=365)

    logger.info(f"\nProcessing {exchange_name}:")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Period:  {start.strftime(config.DEFAULT_CONFIG['output_format']['date_format'])} to {today.strftime(config.DEFAULT_CONFIG['output_format']['date_format'])}")
    
    chunk_results = {}
    process_start_time = time.time()
    rate_limiter = RateLimiter()
    client = None
    
    try:
        # Smaller chunk size for more reliability
        chunk_size = chunk_size or min(config.DEFAULT_CONFIG['batch_size'], 15)
        client = YahooFinanceClient(
            rate_limit_ms=config.DEFAULT_CONFIG['rate_limit_ms'],
            batch_size=chunk_size,
            data_dir=exchange_dir
        )
        await client.__aenter__()
        total_processed = 0
        
        # Process NYSE symbols in smaller batches with progressive delays
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
            
            # Add progressive delay between chunks to avoid overwhelming the API
            if i > 0:
                await asyncio.sleep(min(5, chunk_num * 0.5))
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} symbols)")
            
            for attempt in range(config.DEFAULT_CONFIG['max_retries']):
                try:
                    await rate_limiter.acquire()
                    result = await client.get_history_batch(
                        chunk,
                        start_date=start.strftime(config.DEFAULT_CONFIG['output_format']['date_format']) if not incremental else None,
                        end_date=today.strftime(config.DEFAULT_CONFIG['output_format']['date_format']),
                        incremental=incremental
                    )
                    
                    if result:
                        successful_count = len([v for v in result.values() if v is not None])
                        if successful_count > 0:
                            rate_limiter.report_success()
                        else:
                            rate_limiter.report_failure()
                            
                        chunk_results.update(result)
                        total_processed += len(result)
                        print_progress(total_processed, len(symbols), 
                                      prefix=f'{exchange_name} Progress:')
                        
                        # Only successful symbols count as complete
                        success_rate = successful_count / len(chunk) * 100
                        logger.info(f"Chunk {chunk_num} success rate: {success_rate:.1f}% ({successful_count}/{len(chunk)})")
                        break
                    elif attempt == config.DEFAULT_CONFIG['max_retries'] - 1:
                        logger.error(f"Failed to process chunk {chunk_num}/{total_chunks}")
                        rate_limiter.report_failure()
                    else:
                        # Better backoff with jitter
                        rate_limiter.report_failure()
                        wait_time = (2 ** attempt) + random.random() + 1
                        logger.info(f"Retrying chunk {chunk_num} in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    rate_limiter.report_failure()
                    if attempt == config.DEFAULT_CONFIG['max_retries'] - 1:
                        logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                    wait_time = (2 ** attempt) + random.random() + 1
                    await asyncio.sleep(wait_time)

        if not chunk_results:
            logger.error(f"No successful results for {exchange_name}")
            return {}

        # Get market caps with separate progress tracking
        successful_symbols = [s for s, d in chunk_results.items() if d is not None]
        logger.info(f"Successfully processed {len(successful_symbols)} symbols, fetching market caps")
        
        if successful_symbols:
            active_statuses = await get_market_caps(
                client,
                successful_symbols,
                rate_limiter
            )
        else:
            active_statuses = {}

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

async def update_all_listings(exchanges=None, incremental=False, start_date=None, symbol=None, batch_size=None):
    """Update stock listings for specified exchanges."""
    if symbol:
        logger.info(f"Processing single symbol: {symbol}")
        results = await process_exchange("UNKNOWN", [symbol], start_date, incremental, config.DEFAULT_CONFIG['data_directory'], batch_size)
        return len(results) > 0

    all_exchanges = {
        'NYSE': 'config/nyse-listed.json',
        'NASDAQ': 'config/nasdaq-listed.json',
        'ASX': 'config/asx-listed.json',
        'HKEX': 'config/hkex-listed.json'
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
                json_file = all_exchanges.get(exchange)
                if not json_file or not os.path.exists(json_file):
                    logger.error(f"Missing listings file: {json_file}")
                    results[exchange] = False
                    continue

                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        listings = json.load(f)
                        logger.info(f"Loaded {len(listings)} entries from {json_file}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {json_file}: {e}")
                        results[exchange] = False
                        continue
                    
                # Extract symbols from listings based on file structure
                symbols = []
                for listing in listings:
                    try:
                        # Handle different formats of JSON files
                        if isinstance(listing, str):
                            # Simple string format
                            symbols.append(listing)
                        elif isinstance(listing, dict):
                            # Dictionary format with different key names
                            if 'symbol' in listing:
                                symbols.append(listing['symbol'])
                            elif 'Symbol' in listing:
                                symbols.append(listing['Symbol'])
                            elif 'ACT Symbol' in listing:
                                symbols.append(listing['ACT Symbol'])
                            elif 'ticker' in listing:
                                symbols.append(listing['ticker'])
                            elif 'Ticker' in listing:
                                symbols.append(listing['Ticker'])
                            elif 'Company Name' in listing and exchange == 'NASDAQ':
                                # For NASDAQ, extract ticker from company name if available
                                company_name = listing['Company Name']
                                # Tickers are often at the end, like "Apple Inc. - AAPL"
                                parts = company_name.split(' - ')
                                if len(parts) > 1:
                                    ticker = parts[-1].strip()
                                    if ticker and re.match(r'^[A-Z0-9\.\-]+$', ticker):
                                        symbols.append(ticker)
                    except Exception as e:
                        logger.warning(f"Error parsing listing {listing}: {e}")
                        continue
                
                # Filter out invalid symbols
                valid_symbols = []
                for symbol in symbols:
                    if not isinstance(symbol, str):
                        continue
                        
                    # Clean up symbol
                    clean_symbol = symbol.strip()
                    if not clean_symbol or len(clean_symbol) > config.DEFAULT_CONFIG['symbol_validation']['max_symbol_length']:
                        continue
                    
                    # Skip problematic securities if configured
                    if config.DEFAULT_CONFIG['symbol_validation'].get('skip_special_securities', False):
                        if '.' in clean_symbol and clean_symbol.split('.')[-1] in ['W', 'P', 'U']:
                            continue
                    
                    valid_symbols.append(clean_symbol)
                
                # Remove duplicates and sort
                valid_symbols = sorted(list(set(valid_symbols)))
                
                if not valid_symbols:
                    logger.error(f"No valid symbols found in {json_file}")
                    results[exchange] = False
                    success = False
                    continue
                    
                logger.info(f"Processing {len(valid_symbols)} symbols for {exchange}")
                active_statuses = await process_exchange(
                    exchange, 
                    valid_symbols, 
                    start_date, 
                    incremental, 
                    config.DEFAULT_CONFIG['data_directory'],
                    batch_size
                )
                
                if active_statuses:
                    # Create filtered listings file with only active tickers
                    filtered_file = f"data/{exchange.lower()}_filtered_listings.json"
                    filtered_listings = []
                    
                    for symbol, status in active_statuses.items():
                        # Find the original listing for this symbol
                        listing_info = {}
                        for listing in listings:
                            if not isinstance(listing, dict):
                                continue
                                
                            symbol_match = False
                            if 'symbol' in listing and listing['symbol'] == symbol:
                                symbol_match = True
                                listing_info = listing
                            elif 'Symbol' in listing and listing['Symbol'] == symbol:
                                symbol_match = True
                                listing_info = listing
                            elif 'ACT Symbol' in listing and listing['ACT Symbol'] == symbol:
                                symbol_match = True
                                listing_info = listing
                                
                            if symbol_match:
                                filtered_item = {
                                    'symbol': symbol,
                                    'market_cap': status,
                                    'company_name': listing_info.get('Company Name', listing_info.get('Name', ''))
                                }
                                filtered_listings.append(filtered_item)
                                break
                        
                        # If no listing was found, create minimal entry
                        if not listing_info and status is not None:
                            filtered_listings.append({
                                'symbol': symbol,
                                'market_cap': status
                            })
                    
                    # Save the filtered listings
                    with open(filtered_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_listings, f, indent=4)
                    
                    results[exchange] = True
                    logger.info(f"Created filtered file: {filtered_file} with {len(filtered_listings)} listings")
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

async def schedule_daily_updates():
    """Schedule and run daily updates at configured time."""
    try:
        update_config = config.DEFAULT_CONFIG['update_schedule']
        timezone = pytz.timezone(update_config['timezone'])
        
        while True:
            now = datetime.now(timezone)
            target_time = now.replace(
                hour=update_config['daily_update_hour'],
                minute=update_config['daily_update_minute'],
                second=0,
                microsecond=0
            )
            
            if now >= target_time:
                target_time += timedelta(days=1)
            
            wait_seconds = (target_time - now).total_seconds()
            logger.info(f"Next update scheduled for: {target_time}")
            
            await asyncio.sleep(wait_seconds)
            
            try:
                logger.info("Starting scheduled daily update")
                await update_all_listings(incremental=True)
                logger.info("Completed scheduled daily update")
            except Exception as e:
                logger.error(f"Error during scheduled update: {e}")
            
            # Wait a minute before checking again
            await asyncio.sleep(60)
            
    except asyncio.CancelledError:
        logger.info("Update scheduler stopped")
        raise
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    global loop, scheduler_task
    logger.info("Received interrupt signal, cleaning up...")
    if scheduler_task:
        scheduler_task.cancel()
    if loop and loop.is_running():
        loop.stop()
    raise KeyboardInterrupt

def main(exchanges=None, incremental=False, start_date=None, symbol=None, run_scheduler=False, batch_size=None):
    """Main entry point with proper asyncio handling."""
    global loop, scheduler_task
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if run_scheduler:
            scheduler_task = loop.create_task(schedule_daily_updates())
            return loop.run_forever()
        else:
            return loop.run_until_complete(
                update_all_listings(exchanges, incremental, start_date, symbol, batch_size)
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
        parser.add_argument('--exchanges', nargs='+', choices=['NYSE', 'NASDAQ', 'ASX', 'HKEX'],
                          help='Specify exchanges to update (e.g., NYSE NASDAQ)')
        parser.add_argument('--incremental', action='store_true', 
                          help='Only fetch data since the last available date')
        parser.add_argument('--start-date', type=str, 
                          help=f'Start date for data retrieval (format: {config.DEFAULT_CONFIG["output_format"]["date_format"]})')
        parser.add_argument('--symbol', type=str,
                          help='Update a specific stock symbol')
        parser.add_argument('--scheduler', action='store_true',
                          help='Run as a scheduler for daily updates')
        parser.add_argument('--batch-size', type=int,
                          help='Number of symbols to process in each batch')
        parser.add_argument('--skip-special', action='store_true',
                          help='Skip special securities like warrants and preferred stocks')
        args = parser.parse_args()
        
        if args.skip_special:
            config.DEFAULT_CONFIG['symbol_validation']['skip_special_securities'] = True
            
        if args.start_date and args.incremental:
            logger.error("Cannot use both --start-date and --incremental options")
            exit(1)
            
        main(args.exchanges, args.incremental, args.start_date, args.symbol, args.scheduler, args.batch_size)
        
    except KeyboardInterrupt:
        logger.info("Process terminated by user")
    except Exception as e:
        logger.error(f"Process error: {e}")
        exit(1)
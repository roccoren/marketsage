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
from datetime import date, datetime, timedelta
import re
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from utils.async_utils import YahooFinanceClient

# Enhanced logging configuration
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger first
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)

# Add file handler
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/stock_update.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info("Starting stock update process")

async def process_symbols_async(symbols, start_date=None, days=7, incremental=False, data_dir="data", chunk_size=200, default_start="2024-01-01", exchange_name=None):
    """
    Process stock symbols in parallel using optimized batch processing.
    
    Args:
        symbols: List of stock symbols to process
        start_date: Start date for data retrieval in YYYY-MM-DD format
        days: Default number of days to fetch if incremental=False
        incremental: If True, only fetch data from last available date until now
        data_dir: Directory where stock data is stored
        chunk_size: Number of symbols to process in each chunk
        default_start: Default start date for new stock files when no data exists
        exchange_name: Name of the exchange (NYSE, NASDAQ, etc.)
        
    Returns:
        Dictionary of active symbols with their market cap
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    logger.debug(f"Using data directory: {os.path.abspath(data_dir)}")
    
    today = datetime.today()
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid start date format. Please use YYYY-MM-DD format")
    else:
        start = datetime.strptime(default_start, '%Y-%m-%d') if incremental else today - timedelta(days=days)
    active_statuses = {}
    
    # Process symbols in memory-efficient chunks
    total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(symbols)} symbols in {total_chunks} chunks of size {chunk_size}")
    
    # Create a single client instance for reuse
    client = YahooFinanceClient(data_dir=data_dir, exchange=exchange_name)
    async with client as session:
        tasks = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} symbols)")
            
            # Create task for each chunk
            tasks.append(
                client.get_history_batch(
                    chunk,
                    start_date=start.strftime('%Y-%m-%d') if not incremental else None,
                    end_date=today.strftime('%Y-%m-%d'),
                    incremental=incremental
                )
            )
        
        # Process all chunks concurrently
        chunk_results = await asyncio.gather(*tasks)
        
        # Merge results from all chunks
        active_statuses = {}
        for results in chunk_results:
            if results:
                for symbol, data in results.items():
                    if data is not None:
                        # Get market cap for valid symbols
                        info = await client.get_ticker_info(symbol)
                        market_cap = info['price'].get('marketCap') if info else None
                        active_statuses[symbol] = market_cap
    
    return active_statuses

async def process_hkex(file_path, incremental=False, start_date=None, symbol_filter=None):
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} symbols)")

async def process_nyse_nasdaq(file_path, exchange_name, incremental=False, start_date=None, symbol_filter=None):
    try:
        logging.info(f"Starting {exchange_name} listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        with open(file_path, 'r') as f:
            listings = json.load(f)
            for listing in listings:
                symbol = listing['symbol']
                # If symbol_filter is provided, only process that specific symbol
                if symbol_filter and symbol != symbol_filter:
                    continue
                    
                company_name = listing['company_name']
                # Filter out SPACs, warrants, and similar entities
                if re.search(r'\b(Corporation(s)?|Acquisition(s)?|Warrant(s)?|Unit(s)?|Right(s)?)\b', company_name, re.IGNORECASE):
                    continue
                if symbol.endswith('W') or symbol.endswith('.W'):
                    continue
                symbols_to_check.append(symbol)
                
                # If we found the specific symbol we're looking for, no need to check others
                if symbol_filter and symbol == symbol_filter:
                    break

        logging.info(f"Processing {len(symbols_to_check)} {exchange_name} symbols...")
        if not symbols_to_check:
            logging.warning(f"No matching symbols found in {exchange_name}")
            return False

        active_statuses = await process_symbols_async(symbols_to_check, start_date=start_date, incremental=incremental, exchange_name=exchange_name)
        for listing in listings:
            symbol = listing['symbol']
            if symbol in active_statuses:
                filtered_listings.append({
                    'symbol': symbol,
                    'company_name': listing['company_name'],
                    'market_cap': active_statuses[symbol]
                })

        output_file = f'data/{exchange_name.lower()}_filtered_listings.json'
        with open(output_file, 'w') as f:
            json.dump(filtered_listings, f, indent=4)
        
        logging.info(f"Saved {len(filtered_listings)} {exchange_name} listings to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error processing {exchange_name} listings: {e}")
        return False

async def process_hkex(file_path, incremental=False, start_date=None, symbol_filter=None):
    try:
        logging.info("Starting HKEX listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        with open(file_path, 'r') as f:
            listings = json.load(f)
            
            for listing in listings:
                # Convert to Yahoo Finance format
                yahoo_symbol = f"{listing['symbol']}.HK"
                
                # If symbol_filter is provided, only process that specific symbol
                if symbol_filter and yahoo_symbol != symbol_filter:
                    continue
                    
                symbols_to_check.append(yahoo_symbol)
                logger.debug(f"Added symbol for processing: {yahoo_symbol}")
                
                # If we found the specific symbol we're looking for, no need to check others
                if symbol_filter and yahoo_symbol == symbol_filter:
                    break
        
        logging.info(f"Processing {len(symbols_to_check)} HKEX symbols...")
        active_statuses = await process_symbols_async(symbols_to_check, start_date=start_date, incremental=incremental)

        for listing in listings:
            yahoo_symbol = f"{listing['symbol']}.HK"
            if yahoo_symbol in active_statuses:
                filtered_listings.append({
                    'symbol': listing['symbol'],
                    'company_name': listing['company_name'],
                    'sector': listing.get('sector', ''),
                    'industry': listing.get('industry', ''),
                    'market_cap': active_statuses[yahoo_symbol],
                    'currency': listing.get('currency', 'HKD')
                })
        
        # Sort by symbol for easier reference
        filtered_listings.sort(key=lambda x: x['symbol'])

        with open('data/hkex_filtered_listings.json', 'w') as f:
            json.dump(filtered_listings, f, indent=4)
        
        logging.info(f"Saved {len(filtered_listings)} HKEX listings out of {len(symbols_to_check)} checked symbols")
        return True
    except Exception as e:
        logging.error(f"Error processing HKEX listings: {e}")
        return False

async def process_asx(file_path, incremental=False, start_date=None, symbol_filter=None):
    try:
        logging.info("Starting ASX listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        with open(file_path, 'r') as f:
            listings = json.load(f)
            for listing in listings:
                # Convert to Yahoo Finance format
                yahoo_symbol = f"{listing['symbol']}.AX"
                
                # If symbol_filter is provided, only process that specific symbol
                if symbol_filter and yahoo_symbol != symbol_filter:
                    continue
                    
                symbols_to_check.append(yahoo_symbol)
                logger.debug(f"Added symbol for processing: {yahoo_symbol}")
                
                # If we found the specific symbol we're looking for, no need to check others
                if symbol_filter and yahoo_symbol == symbol_filter:
                    break

        logging.info(f"Processing {len(symbols_to_check)} ASX symbols...")
        active_statuses = await process_symbols_async(symbols_to_check, start_date=start_date, incremental=incremental)

        for listing in listings:
            yahoo_symbol = f"{listing['symbol']}.AX"
            if yahoo_symbol in active_statuses:
                filtered_listings.append({
                    'symbol': listing['symbol'],
                    'company_name': listing['company_name'],
                    'industry': listing.get('industry', ''),
                    'market_cap': active_statuses[yahoo_symbol]
                })

        with open('data/asx_filtered_listings.json', 'w') as f:
            json.dump(filtered_listings, f, indent=4)
        
        logging.info(f"Saved {len(filtered_listings)} ASX listings")
        return True
    except Exception as e:
        logging.error(f"Error processing ASX listings: {e}")
        return False

async def update_all_listings(exchanges=None, incremental=False, start_date=None, symbol=None):
    """
    Update stock listings for specified exchanges.
    
    Args:
        exchanges (list): List of exchanges to update. If None, updates all exchanges.
        incremental (bool): If True, only fetch data since the last update
        start_date (str): Start date for data retrieval in YYYY-MM-DD format
        symbol (str): If provided, only update this specific symbol
    """
    all_exchanges = {
        'NYSE': process_nyse_nasdaq,
        'NASDAQ': process_nyse_nasdaq,
        'HKEX': process_hkex,
        'ASX': process_asx
    }

    # If a specific symbol is provided, determine which exchange to use
    if symbol:
        if symbol.endswith('.HK'):
            exchanges = ['HKEX']
        elif symbol.endswith('.AX'):
            exchanges = ['ASX']
        else:
            # For NYSE/NASDAQ, try both since we don't know which one it's from
            exchanges = ['NYSE', 'NASDAQ']
        logging.info(f"Updating symbol {symbol} on exchanges: {', '.join(exchanges)}")
    
    exchanges_to_process = exchanges if exchanges else all_exchanges.keys()
    logging.info(f"Starting {'incremental' if incremental else 'full'} update of stock listings for: {', '.join(exchanges_to_process)}")
    
    # Ensure data directories exist
    for exchange in exchanges_to_process:
        os.makedirs(f"data/{exchange}", exist_ok=True)

    results = {}
    for exchange in exchanges_to_process:
        if exchange in all_exchanges:
            handler = all_exchanges[exchange]
            # Load and validate filtered listings for each exchange
            try:
                filtered_file = f'data/{exchange.lower()}_filtered_listings.json'
                with open(filtered_file, 'r') as f:
                    listings = json.load(f)
                    
                # Validate listings schema
                required_fields = {'symbol', 'company_name', 'market_cap'}
                if not all(isinstance(listing, dict) and all(field in listing for field in required_fields) for listing in listings):
                    raise ValueError(f"Invalid listing format in {filtered_file}")
                    
                # Process filtered listings for each exchange
                if exchange == 'NYSE':
                    results[exchange] = await handler('data/nyse_filtered_listings.json', 'NYSE', incremental, start_date, symbol)
                elif exchange == 'NASDAQ':
                    results[exchange] = await handler('data/nasdaq_filtered_listings.json', 'NASDAQ', incremental, start_date, symbol)
                elif exchange == 'HKEX':
                    results[exchange] = await handler('data/hkex_filtered_listings.json', incremental, start_date, symbol)
                else:  # ASX
                    results[exchange] = await handler('data/asx_filtered_listings.json', incremental, start_date, symbol)
                    
            except FileNotFoundError:
                logging.error(f"Filtered listings file not found for {exchange}")
                results[exchange] = False
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON format in filtered listings for {exchange}")
                results[exchange] = False
            except ValueError as e:
                logging.error(str(e))
                results[exchange] = False
        else:
            logging.error(f"Unknown exchange: {exchange}")
            results[exchange] = False
    
    # Log overall status
    status = all(results.values())
    if status:
        logging.info(f"Successfully updated stock listings for: {', '.join(exchanges_to_process)}")
    else:
        logging.error("Some exchanges failed to update, check logs for details")
    
    return status

def sync_update_all_listings(exchanges=None, incremental=False, start_date=None, symbol=None):
    """Synchronous wrapper for update_all_listings."""
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(update_all_listings(exchanges, incremental, start_date, symbol))
    except Exception as e:
        logging.error(f"Error during update: {e}")
        return False
    finally:
        # Clean up if we created a new loop
        if loop and not loop.is_running():
            loop.close()

if __name__ == "__main__":
    try:
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # Required for macOS
        
        # Set up command line arguments
        parser = argparse.ArgumentParser(
            description='Update stock listings from various exchanges',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument('--exchanges', nargs='+', choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                          help='Specify exchanges to update (e.g., NYSE NASDAQ). If not specified, updates all exchanges')
        parser.add_argument('--incremental', action='store_true', 
                          help='Only fetch data since the last available date (incremental update)')
        parser.add_argument('--start-date', type=str, 
                          help='Start date for data retrieval (format: YYYY-MM-DD)\nExample: python -u src/update_stocks.py --exchanges HKEX --start-date 2024-01-01')
        parser.add_argument('--symbol', type=str,
                          help='Update a specific stock symbol\nExample: python -u src/update_stocks.py --incremental --symbol AAPL')
        args = parser.parse_args()
        
        if args.start_date and args.incremental:
            logging.error("Cannot use both --start-date and --incremental options")
            exit(1)
        
        # Set start date to 2024-01-01 for new stocks in incremental mode
        if args.incremental and not args.start_date:
            if not args.symbol:
                logger.info("Using 2024-01-01 as start date for new stocks in incremental mode")
        
        # Initial update
        sync_update_all_listings(args.exchanges, args.incremental, args.start_date, args.symbol)
        
        if not args.exchanges:
            logging.info("No specific exchanges selected, will schedule updates for all exchanges")
        else:
            logging.info(f"Scheduling updates for exchanges: {', '.join(args.exchanges)}")

        # Schedule daily updates (run at 00:00 local time)
        # Only schedule if not updating a specific symbol
        if not args.symbol:
            scheduler = BlockingScheduler()
            scheduler.add_job(
                sync_update_all_listings,
                CronTrigger(hour=0, minute=0),
                args=[args.exchanges, True],  # Always use incremental updates for scheduled jobs
                id='stock_update',
                max_instances=1,
                replace_existing=True
            )
            logging.info("Starting scheduler for daily updates")
            scheduler.start()
        
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
    except Exception as e:
        logging.error(f"Scheduler error: {e}")
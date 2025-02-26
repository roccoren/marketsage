import csv
import json
import argparse
import asyncio
import aiohttp
import logging
import time
from datetime import date, datetime, timedelta
import re
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from utils.async_utils import YahooFinanceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_symbols_async(symbols, days=7):
    """
    Process stock symbols in parallel using optimized batch processing.
    """
    today = datetime.today()
    start_date = today - timedelta(days=days)
    active_statuses = {}
    
    # Use optimized client with increased rate limit and batch size
    async with YahooFinanceClient(rate_limit_ms=2000, batch_size=50) as client:
        # Get historical data for all symbols in optimized batches
        results = await client.get_history_batch(
            symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d')
        )
        
        # Process results
        for symbol, data in results.items():
            if data is not None:  # If we got valid data
                # Get market cap for valid symbols
                info = await client.get_ticker_info(symbol)
                market_cap = info['price'].get('marketCap') if info else None
                active_statuses[symbol] = market_cap
    
    return active_statuses

def process_nyse_nasdaq(file_path, exchange_name):
    try:
        logging.info(f"Starting {exchange_name} listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        company_name_header = 'Company Name' if exchange_name == 'NYSE' else 'Security Name'
        symbol_header = 'ACT Symbol' if exchange_name == 'NYSE' else 'Symbol'
        
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                symbol = row[symbol_header]
                company_name = row[company_name_header]

                # Filter out SPACs, warrants, and similar entities
                if re.search(r'\b(Corporation(s)?|Acquisition(s)?|Warrant(s)?|Unit(s)?|Right(s)?)\b', company_name, re.IGNORECASE):
                    continue
                if symbol.endswith('W') or symbol.endswith('.W'):
                    continue

                symbols_to_check.append(symbol)

        logging.info(f"Processing {len(symbols_to_check)} {exchange_name} symbols...")
        active_statuses = asyncio.run(process_symbols_async(symbols_to_check))

        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                symbol = row[symbol_header]
                if symbol in active_statuses:
                    filtered_listings.append({
                        'symbol': symbol,
                        'company_name': row[company_name_header],
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

def process_hkex(file_path):
    try:
        logging.info("Starting HKEX listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            data = list(csv_reader)  # Store all rows
            
            # First pass: collect symbols
            for row in data:
                symbol = row['Symbol']
                if not symbol.endswith('W'):  # Skip warrants
                    symbols_to_check.append(f"{symbol}.HK")  # Add HK suffix for yfinance

        logging.info(f"Processing {len(symbols_to_check)} HKEX symbols...")
        active_statuses = asyncio.run(process_symbols_async(symbols_to_check))

        # Second pass: create filtered listings
        for row in data:
            symbol = row['Symbol']
            yahoo_symbol = f"{symbol}.HK"
            if yahoo_symbol in active_statuses:
                filtered_listings.append({
                    'symbol': symbol,
                    'company_name': row['Company'],
                    'sector': row['Sector'],
                    'industry': row['Industry'],
                    'market_cap': active_statuses[yahoo_symbol],
                    'currency': row['Currency']
                })

        with open('data/hkex_filtered_listings.json', 'w') as f:
            json.dump(filtered_listings, f, indent=4)
        
        logging.info(f"Saved {len(filtered_listings)} HKEX listings")
        return True
    except Exception as e:
        logging.error(f"Error processing HKEX listings: {e}")
        return False

def process_asx(file_path):
    try:
        logging.info("Starting ASX listings processing")
        filtered_listings = []
        symbols_to_check = []
        
        with open(file_path, mode='r', encoding='utf-8') as jsonfile:
            listings = json.load(jsonfile)
            for row in listings:
                symbol = f"{row['symbol']}.AX"  # Add AX suffix for yfinance
                symbols_to_check.append(symbol)

        logging.info(f"Processing {len(symbols_to_check)} ASX symbols...")
        active_statuses = asyncio.run(process_symbols_async(symbols_to_check))

        # Create filtered listings
        for row in listings:
            symbol = f"{row['symbol']}.AX"
            if symbol in active_statuses:
                filtered_listings.append({
                    'symbol': row['symbol'],
                    'company_name': row['companyName'],
                    'industry': row['industry'],
                    'market_cap': active_statuses[symbol]
                })

        with open('data/asx_filtered_listings.json', 'w') as f:
            json.dump(filtered_listings, f, indent=4)
        
        logging.info(f"Saved {len(filtered_listings)} ASX listings")
        return True
    except Exception as e:
        logging.error(f"Error processing ASX listings: {e}")
        return False

def update_all_listings(exchanges=None):
    """
    Update stock listings for specified exchanges.
    Args:
        exchanges (list): List of exchanges to update. If None, updates all exchanges.
    """
    all_exchanges = {
        'NYSE': lambda: process_nyse_nasdaq('data/nyse-listed.csv', 'NYSE'),
        'NASDAQ': lambda: process_nyse_nasdaq('data/nasdaq-listed.csv', 'NASDAQ'),
        'HKEX': lambda: process_hkex('data/hk_stocks_20250218_015522.csv'),
        'ASX': lambda: process_asx('data/asx_valid_listings.json')
    }
    
    exchanges_to_process = exchanges if exchanges else all_exchanges.keys()
    logging.info(f"Starting update of stock listings for: {', '.join(exchanges_to_process)}")
    
    results = {}
    for exchange in exchanges_to_process:
        if exchange in all_exchanges:
            results[exchange] = all_exchanges[exchange]()
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

if __name__ == "__main__":
    try:
        # Set up command line arguments
        parser = argparse.ArgumentParser(description='Update stock listings from various exchanges')
        parser.add_argument('--exchanges', nargs='+', choices=['NYSE', 'NASDAQ', 'HKEX', 'ASX'],
                          help='Specify exchanges to update (e.g., NYSE NASDAQ). If not specified, updates all exchanges.')
        args = parser.parse_args()

        # Initial update
        update_all_listings(args.exchanges)
        
        if not args.exchanges:
            logging.info("No specific exchanges selected, will schedule updates for all exchanges")
        else:
            logging.info(f"Scheduling updates for exchanges: {', '.join(args.exchanges)}")

        # Schedule daily updates (run at 00:00 local time)
        scheduler = BlockingScheduler()
        scheduler.add_job(
            update_all_listings,
            CronTrigger(hour=0, minute=0),
            args=[args.exchanges],
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
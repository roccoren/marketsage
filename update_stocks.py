import csv
import json
import yfinance as yf
import asyncio
import aiohttp
from utils.yfinance_utils import download_with_session
from datetime import date, datetime, timedelta
import re
import logging
from pathlib import Path
import schedule
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_valid_date_range(days=7):
    """Get a valid date range that ends at the most recent market day."""
    end_date = date.today()
    if end_date.year > 2024:
        end_date = date(2024, 2, 23)
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

async def get_market_cap(symbol):
    """Retrieve market cap for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('marketCap')
    except Exception as e:
        logger.error(f"Error fetching market cap for {symbol}: {str(e)}")
        return None

async def fetch_stock_data(session, symbol, start_date, end_date):
    try:
        if symbol.endswith('W') or symbol.endswith('.W'):
            return symbol, False, None

        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None, 
            lambda: download_with_session(symbol, start=start_date, end=end_date, interval="1d")
        )
        
        if data is None or data.empty:
            return symbol, False, None
            
        market_cap = await get_market_cap(symbol)
        return symbol, len(data) > 0, market_cap
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return symbol, False, None
        
async def is_actively_traded_async(symbols, days=7, batch_size=25):
    start_date, end_date = get_valid_date_range(days)
    active_statuses = {}
    market_caps = {}
    
    chunks = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    for chunk in chunks:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in chunk:
                task = fetch_stock_data(session, symbol, start_date, end_date)
                tasks.append(task)
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    symbol = chunk[i]
                    if isinstance(result, tuple):
                        _, is_active, market_cap = result
                        active_statuses[symbol] = is_active
                        market_caps[symbol] = market_cap
                    else:
                        active_statuses[symbol] = False
                        market_caps[symbol] = None
                
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")

    return active_statuses, market_caps

def process_nyse(csv_file_path):
    filtered_listings = []
    symbols_to_check = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            symbol = row['ACT Symbol']
            company_name = row['Company Name']

            if not re.search(r'\b(Corporation(s)?|Acquisition(s)?|Warrant(s)?|Unit(s)?|Right(s)?)\b', company_name, re.IGNORECASE):
                symbols_to_check.append(symbol)

    active_statuses, market_caps = asyncio.run(is_actively_traded_async(symbols_to_check))

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            symbol = row['ACT Symbol']
            if symbol in active_statuses and active_statuses[symbol]:
                filtered_listings.append({
                    'symbol': symbol,
                    'company_name': row['Company Name'],
                    'market_cap': market_caps.get(symbol)
                })
    
    return filtered_listings

def process_hkex(csv_file_path):
    filtered_listings = []
    symbols_to_check = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            symbols_to_check.append(row['Symbol'])

    active_statuses, market_caps = asyncio.run(is_actively_traded_async(symbols_to_check))

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            symbol = row['Symbol']
            if symbol in active_statuses and active_statuses[symbol]:
                filtered_listings.append({
                    'symbol': symbol,
                    'company_name': row['Company'],
                    'market_cap': market_caps.get(symbol),
                    'sector': row.get('Sector', ''),
                    'industry': row.get('Industry', '')
                })
                
    return filtered_listings

def process_asx(json_file_path):
    filtered_listings = []
    symbols_to_check = []
    
    with open(json_file_path, mode='r', encoding='utf-8') as jsonfile:
        listings = json.load(jsonfile)
        for row in listings:
            symbols_to_check.append(f"{row['symbol']}.AX")

    active_statuses, market_caps = asyncio.run(is_actively_traded_async(symbols_to_check))

    with open(json_file_path, mode='r', encoding='utf-8') as jsonfile:
        listings = json.load(jsonfile)
        for row in listings:
            symbol = f"{row['symbol']}.AX"
            if symbol in active_statuses and active_statuses[symbol]:
                filtered_listings.append({
                    'symbol': symbol,
                    'company_name': row['companyName'],
                    'market_cap': market_caps.get(symbol),
                    'industry': row.get('industry', '')
                })
                
    return filtered_listings

def update_stock_listings():
    """Update stock listings for all exchanges."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Process NYSE listings
        nyse_filtered = process_nyse('data/nyse-listed.csv')
        with open(data_dir / 'nyse_filtered_listings.json', 'w') as f:
            json.dump(nyse_filtered, f, indent=4)
        logger.info(f"Processed {len(nyse_filtered)} NYSE listings")

        # Process HKEX listings
        hkex_filtered = process_hkex('data/hk_stocks_20250218_015522.csv')
        with open(data_dir / 'hkex_filtered_listings.json', 'w') as f:
            json.dump(hkex_filtered, f, indent=4)
        logger.info(f"Processed {len(hkex_filtered)} HKEX listings")

        # Process ASX listings
        asx_filtered = process_asx('data/asx_valid_listings.json')
        with open(data_dir / 'asx_filtered_listings.json', 'w') as f:
            json.dump(asx_filtered, f, indent=4)
        logger.info(f"Processed {len(asx_filtered)} ASX listings")

    except Exception as e:
        logger.error(f"Error during stock update process: {str(e)}")
        raise

def run_scheduler():
    """Run the scheduler for daily updates."""
    schedule.every().day.at("00:00").do(update_stock_listings)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    update_stock_listings()  # Run initial update
    run_scheduler()  # Start scheduler for daily updates
"""Specialized utilities for reliable Yahoo Finance data retrieval."""

import yfinance as yf
import pandas as pd
import requests
import logging
import time
import os
import json
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

# Import config settings
from . import yfinance_config as config

logger = logging.getLogger(__name__)

# Global rate limiter to prevent 429 errors
last_request_time = 0
REQUEST_DELAY = 5.0  # Base delay between requests in seconds

def delay_request():
    """Implement rate limiting for Yahoo Finance requests"""
    global last_request_time
    current_time = time.time()
    
    # Add jitter to avoid synchronized requests
    jitter = random.uniform(0.1, 0.5)
    min_delay = REQUEST_DELAY + jitter
    
    if last_request_time > 0:
        elapsed = current_time - last_request_time
        if elapsed < min_delay:
            sleep_time = min_delay - elapsed
            time.sleep(sleep_time)
    
    last_request_time = time.time()

# Create custom session with optimized retry settings
def create_reliable_session() -> requests.Session:
    """Create a requests session with optimal settings for Yahoo Finance."""
    session = requests.Session()
    
    # Configure retry strategy with exponential backoff
    retry_strategy = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )
    
    # Mount the adapter to both http and https
    adapter = HTTPAdapter(max_retries=retry_strategy, 
                          pool_connections=5,  # Reduced connections to avoid overwhelming the API
                          pool_maxsize=10)
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Rotate user agents to avoid blocks
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    
    # Add robust headers
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'DNT': '1',  # Do Not Track
    })
    
    return session

def clean_symbol(symbol: str) -> str:
    """Clean and normalize stock symbol for Yahoo Finance API.
    
    Args:
        symbol: Raw stock symbol from exchange listing
        
    Returns:
        Cleaned symbol compatible with Yahoo Finance
    """
    if not symbol or not isinstance(symbol, str):
        return ""
        
    # Known problematic symbols that need special handling
    known_symbols = {
        'A': 'A',         # Agilent Technologies
        'AA': 'AA',       # Alcoa
        'AAM': 'AAM',     # NYSE special
        'AAME': 'AAME',
        'AAN': 'AAN',
        'T': 'T',         # AT&T
        'F': 'F',         # Ford
        'GM': 'GM',       # General Motors
        'JNJ': 'JNJ',     # Johnson & Johnson
        'IBM': 'IBM',     # IBM
        'MSFT': 'MSFT'    # Microsoft
    }
    
    # Check for known correct symbols first
    if symbol in known_symbols:
        return known_symbols[symbol]
    
    # Remove leading/trailing whitespace
    symbol = symbol.strip()
    
    # Remove leading $ if present
    if symbol.startswith('$'):
        symbol = symbol[1:]
    
    # Handle warrants, units, and preferred stock special cases
    if '.' in symbol:
        parts = symbol.split('.')
        base = parts[0]
        ext = parts[1]
        
        if ext == 'W':  # Warrants
            return f"{base}-WT"
            
        if ext == 'U':  # Units
            return f"{base}-UN"
            
        if ext.startswith('P'):  # Preferred stocks
            return f"{base}-{ext}"
            
        if ext in ['A', 'B', 'C', 'D', 'K'] and len(ext) == 1:
            # Share classes - Yahoo uses dot notation
            return symbol
    
    # Remove invalid characters
    clean_symbol = re.sub(r'[^A-Za-z0-9\.\-]', '', symbol)
    
    return clean_symbol

def fetch_ticker_data(symbol: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None,
                     attempts: int = 3, delay: int = 2) -> Optional[pd.DataFrame]:
    """Fetch ticker data with robust error handling.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        attempts: Number of retry attempts
        delay: Base delay between retries in seconds
        
    Returns:
        DataFrame with historical data or None if failed
    """
    clean_sym = clean_symbol(symbol)
    
    if not clean_sym:
        logger.warning(f"Invalid symbol: {symbol}")
        return None
    
    # Try different symbol formats if needed
    symbol_formats = [clean_sym]
    if '.' not in clean_sym and '-' not in clean_sym:
        symbol_formats.append(f"{clean_sym}.US")
    
    # Setup custom session
    session = create_reliable_session()
    
    # Make sure ticker data directory exists
    cache_dir = os.path.join("data", "yfinance_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Set cache location for this request
        yf.set_tz_cache_location(cache_dir)
    except Exception as e:
        logger.debug(f"Error setting cache location: {e}")
    
    # Optimize date range
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if not start_date:
        # Default to 1 year of data
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Try each format and retry with backoff
    for sym_format in symbol_formats:
        for attempt in range(attempts):
            try:
                # Apply rate limiting
                delay_request()
                
                logger.debug(f"Fetching {sym_format} (attempt {attempt+1}/{attempts})")
                
                # Create ticker with custom session
                ticker = yf.Ticker(sym_format, session=session)
                
                # Get history with a timeout
                data = ticker.history(start=start_date, end=end_date, 
                                      interval='1d', auto_adjust=True)
                
                if not data.empty:
                    logger.info(f"Successfully fetched data for {symbol} as {sym_format}: {len(data)} rows")
                    return data
                    
                else:
                    logger.debug(f"Empty data returned for {symbol} as {sym_format}")
                    
            except Exception as e:
                logger.warning(f"Error fetching {symbol} as {sym_format} (attempt {attempt+1}/{attempts}): {str(e)}")
                if attempt < attempts - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) * delay + random.uniform(1.0, 3.0)
                    logger.debug(f"Waiting {wait_time:.1f}s before retry")
                    time.sleep(wait_time)
    
    logger.error(f"Failed to fetch data for {symbol} after {attempts} attempts")
    return None

def fetch_multiple_tickers(symbols: List[str], start_date: Optional[str] = None,
                          end_date: Optional[str] = None, workers: int = 2, 
                          batch_size: int = 5) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch data for multiple tickers with conservative parallelism.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        workers: Maximum number of parallel workers
        batch_size: Symbols to process in each batch
        
    Returns:
        Dictionary mapping symbols to their data
    """
    results = {}
    total = len(symbols)
    
    # Use very conservative parallelism to avoid rate limits
    workers = min(workers, 1)  # Use sequential processing by default
    batch_size = min(batch_size, 3)  # Smaller batch size
    
    logger.info(f"Fetching data for {total} symbols using {workers} workers")
    
    # Process symbols in batches to avoid overwhelming Yahoo's API
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        # Add significant delay between batches to avoid rate limiting
        if batch_idx > 0:
            delay = 10.0 + random.uniform(2.0, 5.0)  # 10-15 seconds between batches
            logger.debug(f"Waiting {delay:.1f}s between batches")
            time.sleep(delay)
        
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} symbols)")
        
        # Process symbols one at a time in sequential mode if requested
        if workers == 1:
            for symbol in batch:
                try:
                    data = fetch_ticker_data(symbol, start_date, end_date)
                    results[symbol] = data
                    # Add delay between sequential requests
                    time.sleep(1.0 + random.uniform(0.5, 1.5))
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    results[symbol] = None
        else:
            # Process batch with limited parallelism
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                # Submit tasks
                futures = {
                    executor.submit(fetch_ticker_data, symbol, start_date, end_date): symbol 
                    for symbol in batch
                }
                
                # Process results as they complete
                for future in futures:
                    symbol = futures[future]
                    try:
                        data = future.result()
                        results[symbol] = data
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        results[symbol] = None
        
        # Log progress
        completed = batch_idx * batch_size + len(batch)
        success_count = sum(1 for v in list(results.values())[-len(batch):] if v is not None)
        success_rate = (success_count / len(batch)) * 100 if batch else 0
        
        logger.info(f"Batch {batch_idx+1}: {success_rate:.1f}% success rate ({success_count}/{len(batch)})")
        logger.info(f"Overall progress: {completed}/{total} symbols processed")
    
    # Final success rate
    success_count = sum(1 for v in results.values() if v is not None)
    success_rate = (success_count / len(symbols)) * 100 if symbols else 0
    logger.info(f"Completed fetching {len(symbols)} symbols with {success_rate:.1f}% success rate")
    
    return results

def save_ticker_data(symbol: str, data: pd.DataFrame, directory: str) -> bool:
    """Save ticker data to JSON file.
    
    Args:
        symbol: Ticker symbol
        data: DataFrame with historical data
        directory: Directory to save data in
        
    Returns:
        True if saved successfully, False otherwise
    """
    if data is None or data.empty:
        return False
    
    try:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Transform data to desired format
        history = []
        
        for date, row in data.iterrows():
            entry = {
                "date": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            }
            history.append(entry)
        
        # Create final structure
        result = {"history": history}
        
        # Save to file
        file_path = os.path.join(directory, f"{symbol}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {str(e)}")
        return False

def update_exchange_data(exchange: str, symbols: List[str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, data_dir: str = "data",
                       workers: int = 2, batch_size: int = 5) -> Tuple[int, int]:
    """Update data for all symbols in an exchange.
    
    Args:
        exchange: Exchange name (e.g., 'NYSE')
        symbols: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Base directory for data
        workers: Maximum number of parallel workers
        batch_size: Symbols to process in each batch
        
    Returns:
        Tuple of (success_count, total_count)
    """
    if not symbols:
        logger.warning(f"No symbols provided for exchange {exchange}")
        return (0, 0)
    
    # Create exchange directory
    exchange_dir = os.path.join(data_dir, exchange)
    os.makedirs(exchange_dir, exist_ok=True)
    
    logger.info(f"Updating {len(symbols)} symbols for {exchange}")
    logger.info(f"Data will be stored in: {exchange_dir}")
    
    # Start timing
    start_time = time.time()
    
    # Fetch data for all symbols
    results = fetch_multiple_tickers(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        workers=workers,
        batch_size=batch_size
    )
    
    # Save successful results
    success_count = 0
    for symbol, data in results.items():
        if data is not None and not data.empty:
            if save_ticker_data(symbol, data, exchange_dir):
                success_count += 1
    
    # Log summary
    elapsed = time.time() - start_time
    success_rate = (success_count / len(symbols)) * 100 if symbols else 0
    
    logger.info(f"{exchange} update completed in {elapsed:.1f} seconds")
    logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{len(symbols)})")
    
    return (success_count, len(symbols))

def get_ticker_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get basic info for a ticker.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        Dictionary with ticker info or None if failed
    """
    clean_sym = clean_symbol(symbol)
    
    if not clean_sym:
        logger.warning(f"Invalid symbol: {symbol}")
        return None
    
    # Apply rate limiting
    delay_request()
    
    session = create_reliable_session()
    
    try:
        ticker = yf.Ticker(clean_sym, session=session)
        info = ticker.info
        
        # Extract key metrics
        return {
            'symbol': clean_sym,
            'market_cap': info.get('marketCap'),
            'name': info.get('shortName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'exchange': info.get('exchange', ''),
            'currency': info.get('currency', 'USD')
        }
        
    except Exception as e:
        logger.warning(f"Error getting info for {symbol}: {str(e)}")
        return None

def extract_symbols_from_listing(listing_file: str) -> List[str]:
    """Extract ticker symbols from a listing file.
    
    Args:
        listing_file: Path to listing JSON file
        
    Returns:
        List of valid symbols
    """
    try:
        with open(listing_file, 'r', encoding='utf-8') as f:
            listings = json.load(f)
        
        symbols = []
        
        for item in listings:
            if isinstance(item, str):
                symbols.append(item)
            elif isinstance(item, dict):
                # Try different common field names
                for field in ['symbol', 'Symbol', 'ACT Symbol', 'ticker', 'Ticker']:
                    if field in item:
                        symbols.append(item[field])
                        break
        
        # Clean and filter symbols
        valid_symbols = []
        for symbol in symbols:
            if not isinstance(symbol, str):
                continue
                
            # Basic validation
            clean_sym = symbol.strip()
            if clean_sym and len(clean_sym) <= 20:
                valid_symbols.append(clean_sym)
        
        # Remove duplicates
        valid_symbols = list(set(valid_symbols))
        
        return valid_symbols
        
    except Exception as e:
        logger.error(f"Error loading symbols from {listing_file}: {str(e)}")
        return []
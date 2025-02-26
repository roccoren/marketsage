from requests import Session
from requests_cache import CacheMixin, SQLiteCache
import time
import yfinance as yf
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging

class CachedSession(CacheMixin, Session):
    """A requests Session with caching capability."""
    pass

def get_cached_session():
    """
    Returns a cached session for Yahoo Finance API calls.
    Uses SQLite backend to avoid duplicate requests.
    """
    session = CachedSession(
        backend=SQLiteCache("yfinance.cache", expire_after=3600)  # Cache for 1 hour
    )
    return session

def download_with_session(symbol: str, 
                         start: Optional[str] = None, 
                         end: Optional[str] = None, 
                         max_retries: int = 3, 
                         **kwargs) -> Optional[yf.download]:
    """
    Downloads stock data using a cached session with retry logic.
    
    Args:
        symbol: The stock symbol
        start: Start date (optional)
        end: End date (optional)
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments to pass to yf.download
        
    Returns:
        DataFrame with stock data or None if all retries fail
    """
    session = get_cached_session()
    ticker = yf.Ticker(symbol, session=session)
    
    # Debug: Log ticker info
    try:
        info = ticker.info
        logger.debug(f"Ticker info for {symbol}:")
        logger.debug(json.dumps(info, indent=2))
    except Exception as e:
        logger.debug(f"Failed to get ticker info for {symbol}: {str(e)}")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to fetch history for {symbol} (attempt {attempt + 1}/{max_retries})")
            logger.debug(f"Parameters: start={start}, end={end}, kwargs={kwargs}")
            
            data = ticker.history(start=start, end=end, **kwargs)
            
            if data.empty:
                logger.debug(f"Empty data returned for {symbol}")
                if attempt == max_retries - 1:
                    logger.warning(f"{symbol}: No data available in the specified date range")
                    return None
            else:
                logger.debug(f"Successfully fetched data for {symbol}. Shape: {data.shape}")
                return data
                
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.warning(f"Failed to download {symbol} after {max_retries} attempts: {str(e)}")
                return None
            
            # Exponential backoff
            wait_time = (2 ** attempt) + 1  # 1, 3, 5 seconds
            logger.info(f"Attempt {attempt + 1} failed for {symbol}, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
            
    return None
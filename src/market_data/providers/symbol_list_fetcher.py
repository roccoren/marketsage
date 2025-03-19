"""Symbol list-based market data fetcher without market constraints."""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from .enhanced_fetcher import EnhancedMarketFetcher

logger = logging.getLogger(__name__)

# Constants for timeouts and error handling
SAVE_TIMEOUT = 240        # Increased from 180 to 240 seconds for larger batches
BATCH_TIMEOUT = 900      # Keep batch timeout at 15 minutes
SYMBOL_TIMEOUT = 180     # Keep symbol timeout at 3 minutes
MAX_RETRIES = 3          # Maximum number of retries for operation
RETRY_DELAY = 1          # seconds between retries
MAX_PARALLEL_BATCHES = 8  # Increased from 6 to 8 for more parallel processing
EFFECTIVE_BATCH_SIZE = 20  # Increased from 15 to 20 symbols per batch
POINTS_PER_BATCH = 200    # Increased from 100 to 200 points per batch to match InfluxDB batch size

class SymbolListFetcher(EnhancedMarketFetcher):
    """
    Market data fetcher that works with a direct list of symbols without market constraints.
    
    This class extends EnhancedMarketFetcher but allows fetching data for any symbol
    without requiring it to be part of a specific market.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "config/market",
        cache_ttl_days: int = 30,
        max_concurrent: int = 5,
        batch_size: int = 20,
        rate_limit: float = 0.3,
        use_influxdb: bool = True,
        symbols_list: Optional[List[str]] = None,
        force_influxdb: bool = False
    ):
        """
        Initialize the symbol list fetcher.
        
        Args:
            data_dir: Root directory for data storage
            config_dir: Directory for market configuration files
            cache_ttl_days: Days to retain cached data
            max_concurrent: Maximum concurrent operations
            batch_size: Batch size for download operations
            rate_limit: Rate limit in operations per second
            use_influxdb: Whether to use InfluxDB for data storage
            symbols_list: Optional initial list of symbols to use
            force_influxdb: Force InfluxDB storage regardless of config
        """
        # Initialize parent class
        super().__init__(
            data_dir=data_dir,
            config_dir=config_dir,
            cache_ttl_days=cache_ttl_days,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            rate_limit=rate_limit,
            use_influxdb=True  # Always use InfluxDB
        )
        
        # Store concurrency settings for direct access
        self.max_concurrent_operations = max_concurrent
        
        # Store user-provided symbols
        self.user_symbols = set(symbols_list or [])
        
        # Create a dedicated path for symbols not in any market (for fallback)
        self.custom_symbols_path = Path(data_dir) / "custom_symbols"
        self.custom_symbols_path.mkdir(parents=True, exist_ok=True)
        
        # Force InfluxDB usage by modifying config if needed
        if force_influxdb and hasattr(self.cache, 'config'):
            self.cache.config["storage_type"] = "influxdb"
            logger.info("Forced InfluxDB storage enabled")
            
        # Check if InfluxDB is actually available
        self.influxdb_available = hasattr(self.cache, 'client') and self.cache.client is not None
        if not self.influxdb_available:
            logger.warning("InfluxDB is not available. Using file-based fallback storage.")
        else:
            logger.info("InfluxDB is available for data storage")
            
        # Add a cache for downloaded data to avoid redundant downloads
        self._download_cache = {}
    
    def add_symbols(self, symbols: List[str]):
        """Add symbols to the user symbols list."""
        if symbols:
            self.user_symbols.update(symbols)
    
    def get_user_symbols(self) -> List[str]:
        """Get the list of user-defined symbols."""
        return list(self.user_symbols)
    
    def _determine_symbol_market(self, symbol: str) -> str:
        """Determine which market a symbol belongs to."""
        # Check if symbol belongs to any registered market
        for market in self.registry.get_markets():
            if self.registry.validate_symbol(market, symbol):
                return market
        
        # Check for market suffixes
        if symbol.endswith('.HK'):
            return 'hkex'
        elif symbol.endswith('.AX'):
            return 'asx'
            
        # Default to 'custom' if no market is found
        return 'custom'
        
    def _find_symbol_in_df(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """Find how a symbol is represented in a DataFrame with MultiIndex columns."""
        if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
            return None
            
        # Check for exact match
        if symbol in df.columns.levels[0]:
            return symbol
            
        # Check for tuple
        if (symbol,) in df.columns.levels[0]:
            return (symbol,)
            
        # Try uppercase versions
        upper_symbol = symbol.upper()
        if upper_symbol in df.columns.levels[0]:
            return upper_symbol
        if (upper_symbol,) in df.columns.levels[0]:
            return (upper_symbol,)
            
        # Check if it's in any of the column names
        for col in df.columns.levels[0]:
            col_str = col[0] if isinstance(col, tuple) else col
            if symbol.upper() == col_str.upper():
                return col
                
        return None
    
    async def _save_to_storage_with_timeout(self, market: str, symbol: str, date_obj: datetime, data: dict) -> bool:
        """Save data to storage with timeout and retries."""
        for attempt in range(MAX_RETRIES):
            try:
                # Create a task with timeout
                if self.influxdb_available:
                    # Use InfluxDB
                    task = asyncio.create_task(self.cache.set(market, symbol, date_obj, data))
                    done, pending = await asyncio.wait([task], timeout=SAVE_TIMEOUT)
                    
                    if pending:
                        # Timeout occurred
                        for t in pending:
                            t.cancel()
                        logger.warning(f"Timeout when saving {symbol} for {date_obj.strftime('%Y-%m-%d')} to InfluxDB")
                        
                        if attempt < MAX_RETRIES - 1:
                            # Retry after delay
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Progressive backoff
                            continue
                        else:
                            # Save to fallback file on final retry
                            return await self._save_to_fallback_file(market, symbol, date_obj, data)
                    
                    # Check if task completed successfully
                    for t in done:
                        try:
                            t.result()  # Will raise exception if task failed
                            return True
                        except Exception as e:
                            logger.error(f"Error saving to InfluxDB: {str(e)}")
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Progressive backoff
                                continue
                            else:
                                # Save to fallback file on final retry
                                return await self._save_to_fallback_file(market, symbol, date_obj, data)
                else:
                    # Use file fallback directly
                    return await self._save_to_fallback_file(market, symbol, date_obj, data)
                    
            except asyncio.CancelledError:
                # Handle task cancellation
                logger.warning(f"Save operation cancelled for {symbol}")
                return False
                
            except Exception as e:
                logger.error(f"Unexpected error saving {symbol}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Progressive backoff
                    continue
                else:
                    # Try fallback on final retry
                    return await self._save_to_fallback_file(market, symbol, date_obj, data)
        
        return False  # All retries failed
        
    async def _save_to_fallback_file(self, market: str, symbol: str, date_obj: datetime, data: dict) -> bool:
        """Save data to fallback file if InfluxDB fails."""
        try:
            # Determine the directory path based on market
            if market == 'custom':
                symbol_path = self.custom_symbols_path / symbol
            else:
                symbol_path = self.market_paths[market] / symbol
            
            # Create directory if it doesn't exist
            symbol_path.mkdir(parents=True, exist_ok=True)
            
            # Get file path for the date
            date_str = date_obj.strftime('%Y-%m-%d')
            file_path = symbol_path / f"{date_str}.json"
            
            # Save data to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {symbol} data to fallback file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving to fallback file for {symbol}: {str(e)}")
            return False
    
    async def fetch_symbol_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Fetch market data for a list of symbols without market constraints.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data fetching
            end_date: Optional end date (defaults to current date)
            chunk_size: Optional override for batch_size for this operation
        """
        if not symbols:
            logger.warning("No symbols provided for fetching")
            return
        
        # Add these symbols to the user symbols list
        self.add_symbols(symbols)
        
        # Default end date to current date
        end_date = end_date or datetime.now()
        batch_size = chunk_size or self.batch_size
        
        # Break the date range into smaller chunks
        date_chunks = []
        current_start = start_date
        chunk_days = 180  # Process 6 months at a time
        
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            date_chunks.append((current_start, chunk_end))
            current_start = chunk_end + timedelta(days=1)
        
        logger.info(f"Split date range into {len(date_chunks)} chunks")
        
        # Use optimized batch size for more reliable processing
        effective_batch_size = min(batch_size, EFFECTIVE_BATCH_SIZE)
        
        # Group symbols into batches
        batches = []
        for i in range(0, len(symbols), effective_batch_size):
            batch_symbols = symbols[i:i+effective_batch_size]
            batches.append(batch_symbols)
            
        total_batches = len(batches)
        logger.info(f"Fetching data for {len(symbols)} symbols in {total_batches} batches (max {effective_batch_size} symbols per batch)")
        
        successful_symbols = set()  # Use a set to track unique successful symbols
        failed_symbols = set()      # Use a set to track unique failed symbols
        start_time = time.time()
        
        try:
            # Process date chunks sequentially, batches in parallel
            for chunk_num, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
                chunk_start_time = time.time()
                logger.info(f"Processing date chunk {chunk_num}/{len(date_chunks)}: "
                           f"{chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                
                # Process batches with limited parallelism
                batch_semaphore = asyncio.Semaphore(MAX_PARALLEL_BATCHES)
                
                async def process_batch(batch_num, batch_symbols):
                    """Process a single batch with semaphore control"""
                    async with batch_semaphore:
                        return await self._process_batch(batch_num, total_batches, batch_symbols, chunk_start, chunk_end)
                
                # Create tasks for batches in this date chunk
                batch_tasks = []
                for batch_num, batch_symbols in enumerate(batches, 1):
                    task = asyncio.create_task(process_batch(batch_num, batch_symbols))
                    batch_tasks.append(task)
                
                # Wait for all batch tasks to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for batch_num, result in enumerate(batch_results, 1):
                    batch_symbols = batches[batch_num - 1] if batch_num <= len(batches) else []
                    
                    if isinstance(result, Exception):
                        logger.error(f"Batch {batch_num} failed with error: {str(result)}")
                        failed_symbols.update(batch_symbols)
                    else:
                        batch_success, batch_failed = result
                        # Count successful symbols
                        for i in range(batch_success):
                            if i < len(batch_symbols):
                                successful_symbols.add(batch_symbols[i])
                        
                        # Count failed symbols
                        for i in range(batch_success, batch_success + batch_failed):
                            if i < len(batch_symbols):
                                symbol = batch_symbols[i]
                                if symbol not in successful_symbols:
                                    failed_symbols.add(symbol)
                
                chunk_duration = time.time() - chunk_start_time
                logger.info(f"Date chunk {chunk_num} completed in {chunk_duration:.2f} seconds")
                
                # Small delay between chunks
                await asyncio.sleep(1)
            
            total_duration = time.time() - start_time
            logger.info(f"Completed fetching data in {total_duration:.2f} seconds: "
                       f"{len(successful_symbols)} symbols successful, {len(failed_symbols)} symbols failed")
            
        except asyncio.CancelledError:
            logger.warning("Data fetching operation was cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in fetch_symbol_data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _process_batch(self, batch_num: int, total_batches: int, batch_symbols: List[str], 
                            start_date: datetime, end_date: datetime) -> Tuple[int, int]:
        """Process a batch of symbols with streamlined approach."""
        successful = 0
        failed = 0
        
        # Log date range for this chunk
        date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        logger.info(f"Processing batch {batch_num} ({date_range_str})")
        
        # Create a cache key for this date range
        cache_key = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check if we already have this data in cache (simple caching)
        if cache_key in self._download_cache:
            logger.info(f"Using cached data for date range {date_range_str}")
            df = self._download_cache[cache_key]
        else:
            # Download data first - sequential approach
            try:
                df = await self._download_with_retry(batch_symbols, start_date, end_date)
                
                # Cache only if reasonable size and not empty
                if df is not None and not df.empty:
                    # Only keep a small cache with up to 5 entries
                    if len(self._download_cache) >= 5:
                        # Remove oldest entry
                        oldest_key = next(iter(self._download_cache))
                        del self._download_cache[oldest_key]
                        
                    self._download_cache[cache_key] = df
            except Exception as e:
                logger.error(f"Download error: {str(e)}")
                df = None
        
        # If no data available, all symbols failed
        if df is None or df.empty:
            logger.warning(f"No data available for batch {batch_num}")
            return 0, len(batch_symbols)
        
        # Process symbols sequentially to reduce complexity
        for symbol in batch_symbols:
            try:
                success, saved_count = await asyncio.wait_for(
                    self._process_single_symbol(df, symbol),
                    timeout=SYMBOL_TIMEOUT
                )
                
                if success and saved_count > 0:
                    successful += 1
                    # Add to user symbols list to track successful symbols
                    self.add_symbols([symbol])
                else:
                    failed += 1
                    logger.warning(f"Failed to save data for {symbol}")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Processing {symbol} timed out after {SYMBOL_TIMEOUT} seconds")
                failed += 1
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                failed += 1
            
            # Add a small delay between symbols to avoid overwhelming the system
            await asyncio.sleep(0.1)
            
            # Log progress for larger batches
            if len(batch_symbols) > 3 and (successful + failed) % 3 == 0:
                progress = (successful + failed) / len(batch_symbols) * 100
                logger.info(f"Batch {batch_num} progress: {progress:.1f}% ({successful} successful, {failed} failed)")
        
        return successful, failed
    
    async def _process_single_symbol(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, int]:
        """Process and save data for a single symbol."""
        try:
            # Find how the symbol is represented in the DataFrame
            symbol_key = self._find_symbol_in_df(df, symbol)
            
            if symbol_key is None:
                logger.warning(f"No data found for symbol {symbol}")
                return False, 0
                
            # Get symbol data
            symbol_df = df[symbol_key]
            
            # Determine the market for this symbol
            market = self._determine_symbol_market(symbol)
            
            # Save a separate entry for each date in InfluxDB
            saved_dates = 0
            total_dates = len(symbol_df)
            
            # Reduce logging for better performance
            if total_dates > 50:
                logger.info(f"Processing {total_dates} dates for {symbol}")
            
            # Prepare data in batches for more efficient writing to InfluxDB
            data_batch = []
            batch_size = POINTS_PER_BATCH  # Use the increased batch size constant
            
            for date_idx, row in symbol_df.iterrows():
                # Check if the row has valid OHLCV data
                has_valid_data = True
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col not in row or pd.isna(row[col]):
                        has_valid_data = False
                        break
                
                if not has_valid_data:
                    continue
                
                # Format the date string and create date object
                if isinstance(date_idx, pd.Timestamp):
                    date_obj = date_idx.to_pydatetime()
                    date_str = date_idx.strftime('%Y-%m-%d')
                else:
                    try:
                        date_str = str(date_idx)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        continue
                
                # Create data dictionary for InfluxDB
                try:
                    data = {
                        "date": date_str,
                        "symbol": symbol,
                        "market": market,
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": float(row['Volume']),
                        "adjusted_close": float(row['Adj Close']) if 'Adj Close' in row else float(row['Close']),
                    }
                    
                    # Add any additional columns that might be present
                    # Only include important fields to reduce data size
                    for col in row.index:
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                            if col in ['market_cap', 'sector', 'industry'] and not pd.isna(row[col]):
                                val = row[col]
                                if isinstance(val, (int, float)):
                                    data[col.lower()] = float(val)
                                else:
                                    data[col.lower()] = str(val)
                    
                    # Add to batch
                    data_batch.append((date_obj, data))
                    
                    # Process batch if it reaches the batch size
                    if len(data_batch) >= batch_size:
                        batch_saved = await self._save_batch_to_storage(market, symbol, data_batch)
                        saved_dates += batch_saved
                        data_batch = []  # Clear batch after processing
                        
                except Exception as e:
                    logger.error(f"Error preparing data for {symbol} on {date_str}: {str(e)}")
            
            # Process any remaining data
            if data_batch:
                batch_saved = await self._save_batch_to_storage(market, symbol, data_batch)
                saved_dates += batch_saved
            
            if total_dates > 50:
                logger.info(f"Saved {saved_dates}/{total_dates} dates for {symbol}")
            return saved_dates > 0, saved_dates
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return False, 0
            
    async def _save_batch_to_storage(self, market: str, symbol: str, data_batch: List[Tuple[datetime, dict]]) -> int:
        """Save a batch of data points to storage with simplified approach."""
        saved_count = 0
        
        # Check if we can use batch operations in InfluxDB
        if self.influxdb_available and hasattr(self.cache, 'set_batch'):
            try:
                # Prepare batch data
                batch_data = [(market, symbol, date_obj, data) for date_obj, data in data_batch]
                
                # Split into smaller batches to ensure reliable processing
                batch_size = POINTS_PER_BATCH  # Smaller batch size
                
                for i in range(0, len(batch_data), batch_size):
                    sub_batch = batch_data[i:i+batch_size]
                    
                    try:
                        # Simple timeout approach
                        task = asyncio.create_task(self.cache.set_batch(sub_batch))
                        await asyncio.wait_for(task, timeout=SAVE_TIMEOUT)
                        saved_count += len(sub_batch)
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"Batch save failed for {symbol} batch of {len(sub_batch)} points: {str(e)}")
                        
                        # Fall back to individual saves for this batch
                        for market_item, symbol_item, date_obj, data in sub_batch:
                            try:
                                success = await self._save_to_storage_with_timeout(market_item, symbol_item, date_obj, data)
                                if success:
                                    saved_count += 1
                            except Exception as save_error:
                                logger.error(f"Individual save failed: {str(save_error)}")
                
                    # Small delay between batches to avoid overwhelming the database
                    await asyncio.sleep(0.5)
                
                return saved_count
                
            except Exception as e:
                logger.error(f"Batch save error for {symbol}: {str(e)}")
                # Fall through to individual saves
        
        # Fall back to individual saves if batch save is not available or failed
        for date_obj, data in data_batch:
            try:
                success = await self._save_to_storage_with_timeout(market, symbol, date_obj, data)
                if success:
                    saved_count += 1
            except Exception as e:
                logger.error(f"Individual save error: {str(e)}")
        
        return saved_count 
#!/usr/bin/env python3
import json
import os
from datetime import datetime
import pathlib
from typing import Dict, Any, List, Tuple
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools
from tqdm import tqdm
import concurrent.futures
import time
import psutil
import threading

# Setup logging to both file and console
log_file = "market_data_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStats:
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.files_failed = 0
        self.current_batch = 0
        self.total_batches = 0
        self._lock = threading.Lock()

    def update(self, successful: bool):
        with self._lock:
            if successful:
                self.files_processed += 1
            else:
                self.files_failed += 1

    def get_stats(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        files_per_second = self.files_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "processed": self.files_processed,
            "failed": self.files_failed,
            "elapsed_time": elapsed_time,
            "files_per_second": files_per_second,
            "batch_progress": f"{self.current_batch}/{self.total_batches}"
        }

stats = ProcessingStats()

def log_system_stats():
    """Log system resource usage."""
    process = psutil.Process()
    logger.info(
        f"System Stats - CPU: {psutil.cpu_percent()}%, "
        f"Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB, "
        f"Threads: {process.num_threads()}"
    )

def normalize_price(value: float) -> float:
    """Format price values to 4 decimal places."""
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None

def normalize_volume(value: int) -> int:
    """Convert volume to integer."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def normalize_date(date_str: str) -> str:
    """Convert date strings to ISO format YYYY-MM-DD."""
    if not date_str:
        return None
        
    try:
        for fmt in ('%Y-%m-%d', '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y'):
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        logger.warning(f"Could not parse date {date_str} with any known format")
        return date_str
    except Exception as e:
        logger.warning(f"Could not normalize date {date_str}: {e}")
        return date_str

def normalize_json_data(data: Dict[str, Any], market: str, symbol: str) -> Dict[str, Any]:
    """Normalize market data to standard format."""
    try:
        normalized = {
            "open": normalize_price(data.get("open")),
            "high": normalize_price(data.get("high")),
            "low": normalize_price(data.get("low")),
            "close": normalize_price(data.get("close")),
            "volume": normalize_volume(data.get("volume")),
            "adjusted_close": normalize_price(data.get("adjusted_close")),
            "date": normalize_date(data.get("date")),
            "market": market,
            "symbol": symbol,
            "currency": data.get("currency"),
            "trading_status": data.get("trading_status", "active"),
            "metrics": {
                "returns": {
                    "daily": normalize_price(data.get("metrics", {}).get("returns", {}).get("daily")),
                    "intraday_range": normalize_price(data.get("metrics", {}).get("returns", {}).get("intraday_range"))
                },
                "indicators": {
                    "vwap": normalize_price(data.get("metrics", {}).get("indicators", {}).get("vwap")),
                    "turnover": normalize_price(data.get("metrics", {}).get("indicators", {}).get("turnover"))
                }
            },
            "session": {
                "regular": {
                    "start": data.get("session", {}).get("regular", {}).get("start"),
                    "end": data.get("session", {}).get("regular", {}).get("end")
                }
            },
            "metadata": {
                "lot_size": data.get("metadata", {}).get("lot_size"),
                "board_lot": data.get("metadata", {}).get("board_lot"),
                "sector": data.get("metadata", {}).get("sector"),
                "industry": data.get("metadata", {}).get("industry")
            }
        }
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None

def process_file(args: Tuple[str, str, str]) -> bool:
    """Process a single JSON file with market and symbol information."""
    file_path, market, symbol = args
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalized_data = normalize_json_data(data, market, symbol)
        if normalized_data:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            stats.update(True)
            return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        stats.update(False)
    return False

def get_files_to_process(directory: str) -> List[Tuple[str, str, str]]:
    """Get list of files to process with their market and symbol information."""
    logger.info(f"Scanning directory: {directory}")
    files_to_process = []
    directory_path = pathlib.Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    market = directory_path.name
    
    # Use tqdm for directory scanning progress
    symbol_dirs = list(directory_path.glob("**/"))
    with tqdm(symbol_dirs, desc="Scanning directories", unit="dir") as pbar:
        for symbol_dir in pbar:
            if symbol_dir == directory_path:
                continue
                
            symbol = symbol_dir.name
            if not symbol.endswith(('.json', '.JS')):
                json_files = list(symbol_dir.glob("*.json"))
                files_to_process.extend((str(f), market, symbol) for f in json_files)
                pbar.set_postfix({"files_found": len(files_to_process)})
                
    logger.info(f"Found {len(files_to_process)} files to process")
    return files_to_process

def log_progress():
    """Periodically log progress stats."""
    while True:
        time.sleep(10)  # Log every 10 seconds
        current_stats = stats.get_stats()
        logger.info(
            f"Progress - Processed: {current_stats['processed']}, "
            f"Failed: {current_stats['failed']}, "
            f"Rate: {current_stats['files_per_second']:.2f} files/sec, "
            f"Batch: {current_stats['batch_progress']}"
        )
        log_system_stats()

def process_market_directory(directory: str, batch_size: int = 1000, max_workers: int = None) -> None:
    """Process all JSON files in the market data directory structure using parallel processing."""
    try:
        # Start progress logging thread
        progress_thread = threading.Thread(target=log_progress, daemon=True)
        progress_thread.start()

        # Get list of all files to process
        files_to_process = get_files_to_process(directory)
        total_files = len(files_to_process)
        
        if not files_to_process:
            logger.warning(f"No JSON files found in {directory}")
            return
            
        # Use maximum number of cores if not specified
        if max_workers is None:
            max_workers = max(1, cpu_count() - 1)  # Leave one core free
            
        logger.info(f"Starting processing of {total_files} files using {max_workers} workers")
        
        # Calculate total batches for progress tracking
        stats.total_batches = (total_files + batch_size - 1) // batch_size
        
        # Process files in batches using ThreadPoolExecutor
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for i in range(0, total_files, batch_size):
                stats.current_batch += 1
                batch = files_to_process[i:i + batch_size]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_file, batch))
                
                successful = sum(results)
                pbar.update(len(batch))
                logger.info(
                    f"Batch {stats.current_batch}/{stats.total_batches} completed: "
                    f"{successful}/{len(batch)} files processed successfully"
                )
                
        # Log final statistics
        final_stats = stats.get_stats()
        logger.info(
            f"Processing completed - Total processed: {final_stats['processed']}, "
            f"Failed: {final_stats['failed']}, "
            f"Total time: {final_stats['elapsed_time']:.2f}s, "
            f"Average rate: {final_stats['files_per_second']:.2f} files/sec"
        )
                
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {e}")
        raise

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize market data files in parallel')
    parser.add_argument('directory', help='Market data directory to process')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Number of files to process in each batch')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (defaults to CPU count - 1)')
    
    args = parser.parse_args()
    
    try:
        process_market_directory(args.directory, args.batch_size, args.workers)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        final_stats = stats.get_stats()
        logger.info(
            f"Partial results - Processed: {final_stats['processed']}, "
            f"Failed: {final_stats['failed']}"
        )
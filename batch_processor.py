#!/usr/bin/env python3
import json
import os
from datetime import datetime
import pathlib
from typing import Dict, Any, List, Tuple, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, field
from collections import defaultdict
import time
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BatchStats:
    """Track processing statistics for a batch."""
    symbol: str
    files_processed: int = 0
    files_failed: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def processing_rate(self) -> float:
        if self.elapsed_time == 0:
            return 0
        return self.files_processed / self.elapsed_time

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
        return date_str
    except Exception as e:
        logger.warning(f"Could not normalize date {date_str}: {e}")
        return date_str

def normalize_json_data(data: Dict[str, Any], market: str, symbol: str) -> Dict[str, Any]:
    """Normalize market data to standard format."""
    try:
        return {
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
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None

def process_file(args: Tuple[str, str, str, BatchStats]) -> bool:
    """Process a single JSON file."""
    file_path, market, symbol, stats = args
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalized_data = normalize_json_data(data, market, symbol)
        if normalized_data:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            stats.files_processed += 1
            return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        stats.files_failed += 1
    return False

def get_symbol_batches(market_dir: pathlib.Path, max_files: int = 500) -> Iterator[Tuple[str, List[pathlib.Path]]]:
    """Group files by symbol directory, limiting to max_files per batch."""
    if not market_dir.exists():
        raise FileNotFoundError(f"Market directory not found: {market_dir}")
        
    # Get all symbol directories
    symbol_dirs = [d for d in market_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(symbol_dirs)} symbol directories in {market_dir}")
    
    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name
        json_files = list(symbol_dir.glob("*.json"))
        
        # Process files in batches of max_files
        for i in range(0, len(json_files), max_files):
            batch = json_files[i:i + max_files]
            yield symbol, batch

def process_market_data(market: str, workers: int = 4) -> None:
    """Process all JSON files for a given market in symbol-based batches."""
    market_dir = pathlib.Path(f"data/{market}")
    if not market_dir.exists():
        raise FileNotFoundError(f"Market directory not found: {market_dir}")
    
    logger.info(f"Starting processing for market: {market}")
    
    # Process each symbol's files in batches
    for symbol, batch_files in get_symbol_batches(market_dir):
        stats = BatchStats(symbol=symbol)
        logger.info(f"Processing batch for {symbol} ({len(batch_files)} files)")
        
        # Prepare arguments for parallel processing
        process_args = [
            (str(f), market, symbol, stats) for f in batch_files
        ]
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            with tqdm(total=len(batch_files), desc=f"Processing {symbol}") as pbar:
                results = list(executor.map(process_file, process_args))
                pbar.update(len(results))
        
        # Log batch statistics
        logger.info(
            f"Batch complete - Symbol: {symbol}, "
            f"Processed: {stats.files_processed}, "
            f"Failed: {stats.files_failed}, "
            f"Rate: {stats.processing_rate:.2f} files/sec"
        )

def main():
    parser = argparse.ArgumentParser(description='Process market data files by symbol batches')
    parser.add_argument('market', help='Market name (e.g., nyse, nasdaq)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads (default: 4)')
    args = parser.parse_args()
    
    try:
        process_market_data(args.market, args.workers)
        logger.info(f"Processing completed for market: {args.market}")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error processing market {args.market}: {e}")
        raise

if __name__ == "__main__":
    main()
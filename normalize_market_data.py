#!/usr/bin/env python3
import json
import os
from datetime import datetime
import pathlib
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_price(value: float) -> float:
    """Format price values to 4 decimal places."""
    if value is None:
        return None
    return round(float(value), 4)

def normalize_volume(value: int) -> int:
    """Convert volume to integer."""
    if value is None:
        return None
    return int(value)

def normalize_date(date_str: str) -> str:
    """Convert date strings to ISO format YYYY-MM-DD."""
    if not date_str:
        return None
    try:
        # Handle different date formats
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

def process_market_directory(directory: str) -> None:
    """Process all JSON files in the market data directory structure."""
    try:
        directory_path = pathlib.Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        market = directory_path.name
        
        # Walk through all subdirectories
        for symbol_dir in directory_path.glob("**/"):
            if symbol_dir == directory_path:
                continue
                
            symbol = symbol_dir.name
            if not symbol.endswith(('.json', '.JS')):  # Process only symbol directories
                # Process JSON files in the symbol directory
                for json_file in symbol_dir.glob("*.json"):
                    try:
                        logger.info(f"Processing {json_file}")
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        normalized_data = normalize_json_data(data, market, symbol)
                        
                        # Write normalized data back to file
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
                            
                        logger.info(f"Successfully normalized {json_file}")
                    except Exception as e:
                        logger.error(f"Error processing {json_file}: {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python normalize_market_data.py <market_data_directory>")
        sys.exit(1)
        
    market_dir = sys.argv[1]
    process_market_directory(market_dir)
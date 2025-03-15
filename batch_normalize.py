#!/usr/bin/env python3
"""
Batch normalize market data JSON files.
This script processes multiple JSON files in parallel, applying standardization rules.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pytz
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MarketDataNormalizer:
    """Market data normalizer for batch processing."""
    
    MARKET_CONFIG = {
        'asx': {
            'currency': 'AUD',
            'timezone': 'Australia/Sydney',
            'session': {'start': '10:00', 'end': '16:00'},
            'lot_size': 1,
            'board_lot': 1
        },
        'hkex': {
            'currency': 'HKD',
            'timezone': 'Asia/Hong_Kong',
            'session': {'start': '09:30', 'end': '16:00'},
            'lot_size': 100,
            'board_lot': 100
        }
    }

    @staticmethod
    def format_decimal(value: Any, decimal_places: int = 2) -> float:
        """Format decimal with specified precision."""
        if value is None:
            return None
        try:
            dec = Decimal(str(float(value)))
            rounded = dec.quantize(Decimal(f'0.{"0" * decimal_places}'), rounding=ROUND_HALF_UP)
            return float(rounded)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def format_integer(value: Any) -> int:
        """Format value as integer."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate trading metrics."""
        metrics = {
            'returns': {'daily': None, 'intraday_range': None},
            'indicators': {'vwap': None, 'turnover': None}
        }
        
        try:
            # Calculate returns
            if all(data.get(k) for k in ['open', 'close']):
                daily_return = ((data['close'] - data['open']) / data['open']) * 100
                metrics['returns']['daily'] = self.format_decimal(daily_return)

            if all(data.get(k) for k in ['high', 'low']):
                intraday_range = ((data['high'] - data['low']) / data['low']) * 100
                metrics['returns']['intraday_range'] = self.format_decimal(intraday_range)

            # Calculate indicators
            if all(data.get(k) for k in ['high', 'low', 'close']):
                vwap = (data['high'] + data['low'] + data['close']) / 3
                metrics['indicators']['vwap'] = self.format_decimal(vwap)

            if all(data.get(k) for k in ['close', 'volume']):
                turnover = float(data['volume']) * float(data['close'])
                metrics['indicators']['turnover'] = self.format_decimal(turnover, 2)

        except Exception as e:
            logging.warning(f"Error calculating metrics: {str(e)}")

        return metrics

    def get_session_times(self, date_str: str, market: str) -> Dict[str, str]:
        """Get market session times in ISO format."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            market_config = self.MARKET_CONFIG.get(market.lower(), self.MARKET_CONFIG['asx'])
            timezone = pytz.timezone(market_config['timezone'])
            
            start_time = datetime.strptime(market_config['session']['start'], '%H:%M').time()
            end_time = datetime.strptime(market_config['session']['end'], '%H:%M').time()
            
            start_dt = timezone.localize(datetime.combine(date, start_time))
            end_dt = timezone.localize(datetime.combine(date, end_time))
            
            return {
                'start': start_dt.isoformat(),
                'end': end_dt.isoformat()
            }
        except Exception as e:
            logging.warning(f"Error generating session times: {str(e)}")
            return {'start': None, 'end': None}

    def normalize_file(self, file_path: Path) -> Tuple[Path, bool]:
        """Normalize a single JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            market = file_path.parts[-3].lower()
            symbol = file_path.parts[-2]
            market_config = self.MARKET_CONFIG.get(market, self.MARKET_CONFIG['asx'])

            normalized = {
                "open": self.format_decimal(data.get("open")),
                "high": self.format_decimal(data.get("high")),
                "low": self.format_decimal(data.get("low")),
                "close": self.format_decimal(data.get("close")),
                "volume": self.format_integer(data.get("volume")),
                "adjusted_close": self.format_decimal(data.get("adjusted_close")),
                "date": data.get("date"),
                "market": market,
                "symbol": symbol,
                "currency": market_config['currency'],
                "trading_status": "active",
                "metrics": self.calculate_metrics(data),
                "session": {
                    "regular": self.get_session_times(data.get("date"), market)
                },
                "metadata": {
                    "lot_size": market_config['lot_size'],
                    "board_lot": market_config['board_lot'],
                    "sector": None,
                    "industry": None
                }
            }

            with open(file_path, 'w') as f:
                json.dump(normalized, f, indent=2)

            return file_path, True
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return file_path, False

def process_directory(base_dir: str, max_workers: int = None) -> List[Tuple[Path, bool]]:
    """Process all JSON files in directory structure."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Directory not found: {base_dir}")

    normalizer = MarketDataNormalizer()
    files_to_process = []

    # Find all JSON files
    for market_dir in base_path.glob('*'):
        if not market_dir.is_dir():
            continue
        for symbol_dir in market_dir.glob('*'):
            if not symbol_dir.is_dir():
                continue
            files_to_process.extend(symbol_dir.glob('*.json'))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(normalizer.normalize_file, file_path): file_path
            for file_path in files_to_process
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                if result[1]:
                    logging.info(f"Successfully processed: {result[0]}")
                else:
                    logging.error(f"Failed to process: {result[0]}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                results.append((file_path, False))

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch normalize market data JSON files")
    parser.add_argument("directory", help="Base directory containing market data files")
    parser.add_argument("--workers", type=int, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    try:
        results = process_directory(args.directory, args.workers)
        success = sum(1 for _, ok in results if ok)
        total = len(results)
        print(f"\nNormalization complete: {success}/{total} files processed successfully")
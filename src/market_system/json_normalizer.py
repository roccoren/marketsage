#!/usr/bin/env python3
"""
JSON normalizer for market data files.
Standardizes format and precision of market data JSON files.
"""

import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime
import logging
from typing import Dict, Any, Union
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MarketDataNormalizer:
    """Handles normalization of market data JSON files."""
    
    PRICE_DECIMAL_PLACES = 2  # Updated to 2 decimal places as per requirements
    
    # Market specific configurations
    MARKET_CONFIG = {
        'asx': {
            'currency': 'AUD',
            'timezone': 'Australia/Sydney',
            'session': {
                'start': '10:00',
                'end': '16:00'
            },
            'lot_size': 1,
            'board_lot': 1
        },
        'hkex': {
            'currency': 'HKD',
            'timezone': 'Asia/Hong_Kong',
            'session': {
                'start': '09:30',
                'end': '16:00'
            },
            'lot_size': 100,
            'board_lot': 100
        }
    }
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize normalizer with base directory path."""
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")

    def format_decimal(self, value: Union[float, int, str, None], decimal_places: int = None) -> float:
        """Format decimal value to specified precision."""
        if decimal_places is None:
            decimal_places = self.PRICE_DECIMAL_PLACES
            
        if value is None:
            return None
        try:
            # Convert to Decimal for precise rounding
            dec = Decimal(str(float(value)))
            rounded = dec.quantize(Decimal(f'0.{"0" * decimal_places}'), rounding=ROUND_HALF_UP)
            return float(rounded)
        except (ValueError, TypeError, InvalidOperation) as e:
            logging.warning(f"Failed to format decimal value: {value} - {str(e)}")
            return None

    def format_integer(self, value: Union[float, int, str, None]) -> int:
        """Format value as integer."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logging.warning(f"Failed to format integer value: {value}")
            return None

    def format_date(self, date_str: str) -> str:
        """Format date string to ISO format."""
        if not date_str:
            return None
        try:
            # Handle multiple date formats
            for fmt in ('%Y-%m-%d', '%Y%m%d', '%d/%m/%Y'):
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_str}")
        except Exception as e:
            logging.warning(f"Failed to format date: {date_str} - {str(e)}")
            return None

    def calculate_returns(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate return metrics."""
        try:
            daily = None
            if data.get('open') and data.get('close'):
                daily = ((data['close'] - data['open']) / data['open']) * 100

            intraday_range = None
            if data.get('high') and data.get('low'):
                intraday_range = ((data['high'] - data['low']) / data['low']) * 100

            return {
                'daily': self.format_decimal(daily),
                'intraday_range': self.format_decimal(intraday_range)
            }
        except Exception as e:
            logging.warning(f"Failed to calculate returns: {str(e)}")
            return {'daily': None, 'intraday_range': None}

    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate trading indicators."""
        try:
            # Calculate VWAP (Volume Weighted Average Price)
            if all(data.get(x) for x in ['high', 'low', 'close']):
                vwap = (data['high'] + data['low'] + data['close']) / 3
            else:
                vwap = None

            # Calculate turnover
            turnover = None
            if data.get('volume') is not None and data.get('close') is not None:
                turnover = float(data['volume']) * float(data['close'])

            return {
                'vwap': self.format_decimal(vwap),
                'turnover': self.format_decimal(turnover, decimal_places=2)
            }
        except Exception as e:
            logging.warning(f"Failed to calculate indicators: {str(e)}")
            return {'vwap': None, 'turnover': None}

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
            logging.warning(f"Failed to get session times: {str(e)}")
            return {'start': None, 'end': None}

    def get_metadata(self, market: str, symbol: str) -> Dict[str, Any]:
        """Get metadata for symbol."""
        market_config = self.MARKET_CONFIG.get(market.lower(), self.MARKET_CONFIG['asx'])
        return {
            'lot_size': market_config['lot_size'],
            'board_lot': market_config['board_lot'],
            'sector': None,
            'industry': None
        }

    def normalize_data(self, data: Dict[str, Any], market: str, symbol: str) -> Dict[str, Any]:
        """Normalize market data according to standard format."""
        date = self.format_date(data.get("date"))
        
        # Base price and volume data
        normalized = {
            "open": self.format_decimal(data.get("open")),
            "high": self.format_decimal(data.get("high")),
            "low": self.format_decimal(data.get("low")),
            "close": self.format_decimal(data.get("close")),
            "volume": self.format_integer(data.get("volume")),
            "adjusted_close": self.format_decimal(data.get("adjusted_close")),
            "date": date,
            "market": market.lower(),
            "symbol": symbol,
            "currency": self.MARKET_CONFIG.get(market.lower(), {}).get('currency'),
            "trading_status": "active"  # Default to active
        }
        
        # Add calculated metrics
        returns = self.calculate_returns(normalized)
        indicators = self.calculate_indicators(normalized)
        
        normalized["metrics"] = {
            "returns": returns,
            "indicators": indicators
        }
        
        # Add session information
        if date:
            normalized["session"] = {
                "regular": self.get_session_times(date, market)
            }
        
        # Add metadata
        normalized["metadata"] = self.get_metadata(market, symbol)
        
        # Remove any None values
        return {k: v for k, v in normalized.items() if v is not None}

    def process_file(self, file_path: Path) -> None:
        """Process a single JSON file."""
        try:
            # Extract market and symbol from path
            market = file_path.parts[-3]
            symbol = file_path.parts[-2]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            normalized_data = self.normalize_data(data, market, symbol)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Normalized {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {str(e)}")

    def process_all_files(self) -> None:
        """Process all JSON files in the data directory structure."""
        try:
            for market_dir in self.base_dir.glob('*'):
                if not market_dir.is_dir():
                    continue
                    
                for symbol_dir in market_dir.glob('*'):
                    if not symbol_dir.is_dir():
                        continue
                        
                    for json_file in symbol_dir.glob('*.json'):
                        self.process_file(json_file)
                        
        except Exception as e:
            logging.error(f"Failed to process files: {str(e)}")

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Normalize market data JSON files.')
    parser.add_argument('base_dir', type=str, help='Base directory containing market data')
    args = parser.parse_args()
    
    normalizer = MarketDataNormalizer(args.base_dir)
    normalizer.process_all_files()

if __name__ == '__main__':
    main()
"""
Core market data normalizer.
Handles standardization of market data formats.
"""

from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime
import logging
from typing import Dict, Any, Union, Optional
import pytz
import json

from .base_processor import BaseProcessor, MarketDataError

logger = logging.getLogger(__name__)

class MarketDataNormalizer(BaseProcessor):
    """Base normalizer for market data files."""
    
    # Default decimal precision for price values
    PRICE_DECIMAL_PLACES = 4
    
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
        },
        'nyse': {
            'currency': 'USD',
            'timezone': 'America/New_York',
            'session': {
                'start': '09:30',
                'end': '16:00'
            },
            'lot_size': 1,
            'board_lot': 1
        }
    }
    
    def format_decimal(self, value: Union[float, int, str, None], decimal_places: int = None) -> Optional[float]:
        """
        Format a value to a specific number of decimal places.
        
        Args:
            value: Value to format
            decimal_places: Number of decimal places (defaults to PRICE_DECIMAL_PLACES)
            
        Returns:
            float: Formatted value
        """
        if value is None:
            return None
            
        if decimal_places is None:
            decimal_places = self.PRICE_DECIMAL_PLACES
            
        try:
            # Use Decimal for precise rounding
            decimal_value = Decimal(str(float(value)))
            rounded = decimal_value.quantize(
                Decimal('0.' + '0' * decimal_places),
                rounding=ROUND_HALF_UP
            )
            return float(rounded)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning(f"Could not format decimal {value}: {e}")
            return None
    
    def format_integer(self, value: Union[float, int, str, None]) -> Optional[int]:
        """
        Format a value as an integer.
        
        Args:
            value: Value to format
            
        Returns:
            int: Formatted value
        """
        if value is None:
            return None
            
        try:
            # Round to nearest integer
            decimal_value = Decimal(str(float(value)))
            rounded = decimal_value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            return int(rounded)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning(f"Could not format integer {value}: {e}")
            return None
    
    def format_date(self, date_str: str) -> str:
        """
        Format date string to ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string to format
            
        Returns:
            str: Formatted date string
        """
        return self.normalize_date(date_str)
    
    def calculate_returns(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate returns from price data.
        
        Args:
            data: Market data dictionary with price information
            
        Returns:
            Dict[str, float]: Dictionary of calculated returns
        """
        returns = {}
        
        try:
            # Daily return calculation
            if all(key in data for key in ['open', 'close']):
                open_price = self.format_decimal(data['open'])
                close_price = self.format_decimal(data['close'])
                
                if open_price and open_price > 0:
                    daily_return = (close_price - open_price) / open_price
                    returns['daily_return'] = self.format_decimal(daily_return, 6)
                    returns['daily_return_pct'] = self.format_decimal(daily_return * 100, 4)
        except Exception as e:
            logger.warning(f"Failed to calculate returns: {e}")
        
        return returns
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate basic indicators from price data.
        
        Args:
            data: Market data dictionary with price information
            
        Returns:
            Dict[str, float]: Dictionary of calculated indicators
        """
        indicators = {}
        
        try:
            # Calculate price range
            if all(key in data for key in ['high', 'low']):
                high = self.format_decimal(data['high'])
                low = self.format_decimal(data['low'])
                
                if high is not None and low is not None and low > 0:
                    price_range = high - low
                    range_pct = price_range / low
                    
                    indicators['price_range'] = price_range
                    indicators['range_pct'] = self.format_decimal(range_pct * 100, 4)
                    
            # Calculate volume indicators
            if all(key in data for key in ['close', 'volume']):
                close = self.format_decimal(data['close'])
                volume = self.format_integer(data['volume'])
                
                if close is not None and close > 0 and volume is not None:
                    indicators['value_traded'] = self.format_decimal(close * volume)
        except Exception as e:
            logger.warning(f"Failed to calculate indicators: {e}")
        
        return indicators
    
    def get_session_times(self, date_str: str, market: str) -> Dict[str, str]:
        """
        Get market session times for a specific date.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD)
            market: Market code (e.g., 'asx', 'hkex')
            
        Returns:
            Dict[str, str]: Dictionary with session times
        """
        if market not in self.MARKET_CONFIG:
            return {}
            
        market_config = self.MARKET_CONFIG[market]
        
        # Get timezone for the market
        timezone = pytz.timezone(market_config['timezone'])
        
        # Parse date
        try:
            base_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Add session times
            session_start = datetime.strptime(
                f"{date_str} {market_config['session']['start']}",
                '%Y-%m-%d %H:%M'
            ).replace(tzinfo=timezone)
            
            session_end = datetime.strptime(
                f"{date_str} {market_config['session']['end']}",
                '%Y-%m-%d %H:%M'
            ).replace(tzinfo=timezone)
            
            return {
                'session_start': session_start.isoformat(),
                'session_end': session_end.isoformat()
            }
        except Exception as e:
            logger.warning(f"Error processing session times for {date_str}: {e}")
            return {}
    
    def get_metadata(self, market: str, symbol: str) -> Dict[str, Any]:
        """
        Get market metadata for a symbol.
        
        Args:
            market: Market code (e.g., 'asx', 'hkex')
            symbol: Symbol code
            
        Returns:
            Dict[str, Any]: Market metadata
        """
        metadata = {
            'market': market,
            'symbol': symbol
        }
        
        if market in self.MARKET_CONFIG:
            metadata.update({
                'currency': self.MARKET_CONFIG[market]['currency'],
                'timezone': self.MARKET_CONFIG[market]['timezone'],
                'lot_size': self.MARKET_CONFIG[market]['lot_size']
            })
        
        return metadata
    
    def normalize_data(self, data: Dict[str, Any], market: str, symbol: str) -> Dict[str, Any]:
        """
        Normalize market data to standard format.
        
        Args:
            data: Raw market data dictionary
            market: Market code
            symbol: Symbol code
            
        Returns:
            Dict[str, Any]: Normalized data
        """
        if not data:
            return {}
            
        normalized = {
            # Basic price data
            'open': self.format_decimal(data.get('open')),
            'high': self.format_decimal(data.get('high')),
            'low': self.format_decimal(data.get('low')),
            'close': self.format_decimal(data.get('close')),
            'volume': self.format_integer(data.get('volume')),
            'adjusted_close': self.format_decimal(data.get('adjusted_close')),
            
            # Date and metadata
            'date': self.format_date(data.get('date')),
            'market': market,
            'symbol': symbol
        }
        
        # Add market metadata
        normalized.update(self.get_metadata(market, symbol))
        
        # Add session times
        if normalized['date']:
            normalized.update(self.get_session_times(normalized['date'], market))
            
        # Calculate returns and indicators
        returns = self.calculate_returns(normalized)
        indicators = self.calculate_indicators(normalized)
        
        normalized.update(returns)
        normalized.update(indicators)
        
        # Remove None values
        return {k: v for k, v in normalized.items() if v is not None}
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process and normalize a market data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            bool: True if successful
        """
        try:
            # Extract market and symbol from file path
            market = file_path.parent.parent.name
            symbol = file_path.parent.name
            
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Normalize data
            normalized = self.normalize_data(data, market, symbol)
            
            # Write normalized data back
            with open(file_path, 'w') as f:
                json.dump(normalized, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error normalizing {file_path}: {e}")
            return False
    
    def process_all_files(self) -> Dict[str, int]:
        """
        Process all market data files.
        
        Returns:
            Dict[str, int]: Processing statistics
        """
        stats = {
            'processed': 0,
            'failed': 0,
            'total': 0
        }
        
        for market_dir in self.base_path.iterdir():
            if not market_dir.is_dir():
                continue
                
            market = market_dir.name
            logger.info(f"Processing market: {market}")
            
            for symbol_dir in market_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                    
                symbol = symbol_dir.name
                logger.info(f"Processing symbol: {symbol}")
                
                for file_path in symbol_dir.glob('*.json'):
                    stats['total'] += 1
                    
                    if self.process_file(file_path):
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1
        
        logger.info(f"Normalization complete. Processed: {stats['processed']}, " 
                    f"Failed: {stats['failed']}, Total: {stats['total']}")
        return stats 
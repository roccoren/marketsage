"""
Base processor class for market data processing.
Provides common functionality for market data processors.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Union, Any, Generator
import pandas as pd
from datetime import datetime
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define JSON schema for market data validation
MARKET_DATA_SCHEMA = {
    "type": "object",
    "required": ["open", "high", "low", "close", "volume", "adjusted_close", "date"],
    "properties": {
        "open": {"type": "number", "minimum": 0},
        "high": {"type": "number", "minimum": 0},
        "low": {"type": "number", "minimum": 0},
        "close": {"type": "number", "minimum": 0},
        "volume": {"type": "number", "minimum": 0},
        "adjusted_close": {"type": "number", "minimum": 0},
        "date": {"type": "string", "format": "date"},
        "market": {"type": "string"},
        "symbol": {"type": "string"}
    }
}

class MarketDataError(Exception):
    """Custom exception for market data processing errors."""
    pass

class BaseProcessor:
    """
    Base class for all market data processors.
    
    Provides common functionality for file handling, validation,
    and basic data processing operations.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the processor with the base path for market data.
        
        Args:
            base_path: Path to the market data directory
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise MarketDataError(f"Base path does not exist: {self.base_path}")
    
    def validate_json_file(self, file_path: Path) -> Dict:
        """
        Validate and load a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dict: Validated JSON data
            
        Raises:
            MarketDataError: If validation fails
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            validate(instance=data, schema=MARKET_DATA_SCHEMA)
            return data
        except json.JSONDecodeError as e:
            raise MarketDataError(f"Invalid JSON in file {file_path}: {e}")
        except ValidationError as e:
            raise MarketDataError(f"Schema validation failed for {file_path}: {e}")
        except Exception as e:
            raise MarketDataError(f"Error processing file {file_path}: {e}")
    
    def _read_json_chunk(self, file_path: Path, chunk_size: int = 10000) -> Generator[Dict, None, None]:
        """
        Read a JSON file in chunks to manage memory usage.
        
        Args:
            file_path: Path to the JSON file
            chunk_size: Number of records per chunk
            
        Yields:
            Dict: Chunk of JSON data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # For list-type JSON files
            if isinstance(data, list):
                for i in range(0, len(data), chunk_size):
                    yield data[i:i+chunk_size]
            # For single objects
            else:
                yield data
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise MarketDataError(f"Failed to read JSON file {file_path}: {e}")
    
    def get_market_symbols(self, market_name: str) -> List[str]:
        """
        Get list of symbols for a market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            List[str]: List of symbol names
        """
        market_dir = self.base_path / market_name
        if not market_dir.exists():
            raise MarketDataError(f"Market directory does not exist: {market_dir}")
            
        return [d.name for d in market_dir.iterdir() if d.is_dir()]
    
    def normalize_date(self, date_str: str) -> str:
        """
        Convert date strings to ISO format YYYY-MM-DD.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            str: Normalized date string
        """
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
            
    def normalize_price(self, value: Union[float, int, str, None], decimal_places: int = 4) -> Optional[float]:
        """
        Format price values to specified decimal places.
        
        Args:
            value: Price value to normalize
            decimal_places: Number of decimal places to round to
            
        Returns:
            float: Normalized price value
        """
        if value is None:
            return None
        try:
            return round(float(value), decimal_places)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not normalize price {value}: {e}")
            return None
            
    def normalize_volume(self, value: Union[int, float, str, None]) -> Optional[int]:
        """
        Convert volume to integer.
        
        Args:
            value: Volume value to normalize
            
        Returns:
            int: Normalized volume value
        """
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not normalize volume {value}: {e}")
            return None 
"""
Market data processor module for handling market system JSON data files.
Implements validation and processing of market data with pandas integration.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from jsonschema import validate, ValidationError

from .core.base_processor import BaseProcessor, MarketDataError

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

class MarketDataProcessor(BaseProcessor):
    """
    Processes market system data from JSON files.
    
    Handles validation, loading, and processing of market data files
    with proper error handling and data integrity checks.
    """
    
    def process_market_data(self, market_name: str) -> pd.DataFrame:
        """
        Process all market data for a specific market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            pd.DataFrame: Processed market data
        """
        market_dir = self.base_path / market_name
        if not market_dir.exists():
            raise MarketDataError(f"Market directory does not exist: {market_dir}")
        
        all_data = []
        
        symbols = self.get_market_symbols(market_name)
        
        for symbol in symbols:
            symbol_dir = market_dir / symbol
            if not symbol_dir.is_dir():
                continue
                
            for file_path in symbol_dir.glob('*.json'):
                try:
                    data = self.validate_json_file(file_path)
                    
                    # Add symbol if not present
                    if 'symbol' not in data:
                        data['symbol'] = symbol
                    
                    # Add market if not present
                    if 'market' not in data:
                        data['market'] = market_name
                    
                    # Normalize date
                    data['date'] = self.normalize_date(data['date'])
                    
                    all_data.append(data)
                except MarketDataError as e:
                    logger.warning(f"Skipping file {file_path}: {e}")
        
        if not all_data:
            raise MarketDataError(f"No valid data found for market: {market_name}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set index
        df.set_index(['date', 'symbol'], inplace=True)
        
        # Sort by date and symbol
        df.sort_index(inplace=True)
        
        return df
    
    def get_market_summary(self, market_name: str) -> Dict:
        """
        Get summary statistics for a market.
        
        Args:
            market_name: Name of the market
            
        Returns:
            Dict: Summary statistics
        """
        try:
            df = self.process_market_data(market_name)
            
            summary = {
                'market': market_name,
                'symbols': df.index.get_level_values('symbol').nunique(),
                'days': df.index.get_level_values('date').nunique(),
                'start_date': df.index.get_level_values('date').min().strftime('%Y-%m-%d'),
                'end_date': df.index.get_level_values('date').max().strftime('%Y-%m-%d'),
                'avg_price': df['close'].mean(),
                'avg_volume': df['volume'].mean(),
                'total_records': len(df)
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {
                'market': market_name,
                'error': str(e)
            }

    def validate_json_file(self, file_path: Path) -> Dict:
        """
        Validate a JSON file against the market data schema.
        
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
            raise MarketDataError(f"Invalid JSON in {file_path}: {str(e)}")
        except ValidationError as e:
            raise MarketDataError(f"Schema validation failed for {file_path}: {str(e)}")
        except Exception as e:
            raise MarketDataError(f"Error processing {file_path}: {str(e)}")
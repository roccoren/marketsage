import json
import os
import pandas as pd
from typing import Dict, Optional
import logging
from datetime import datetime

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Add file handler
    file_handler = logging.FileHandler('logs/exchange_data.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ExchangeDataLoader:
    """Loads and processes market data from different exchanges."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            base_dir: Base directory containing exchange data folders
        """
        self.base_dir = base_dir
        self.exchanges = ['ASX', 'NYSE', 'NASDAQ', 'HKEX']
        
    def load_stock_data(self, exchange: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific stock.
        
        Args:
            exchange: Exchange name (ASX, NYSE, NASDAQ, HKEX)
            symbol: Stock symbol
            
        Returns:
            DataFrame with historical data or None if data is invalid/missing
        """
        try:
            # Construct file path
            file_path = os.path.join(self.base_dir, exchange, f"{symbol}.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                return None
                
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded JSON data from {file_path}")
                
            # Validate data structure
            if not self._validate_data_structure(data):
                logger.error(f"Invalid data structure in {file_path}")
                return None
                
            # Convert historical data to DataFrame
            df = pd.DataFrame(data['history'])
            logger.debug(f"Created DataFrame with shape: {df.shape}")
            
            # Process and validate DataFrame
            return self._process_dataframe(df)
            
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading data for {symbol} from {exchange}: {str(e)}", exc_info=True)
            return None
            
    def _validate_data_structure(self, data: Dict) -> bool:
        """Validate the structure of loaded JSON data."""
        required_fields = ['code', 'history']
        if not all(field in data for field in required_fields):
            return False
            
        if not isinstance(data['history'], list) or not data['history']:
            return False
            
        # Validate first history entry has required fields
        required_history_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        first_entry = data['history'][0]
        return all(field in first_entry for field in required_history_fields)
        
    def _process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process and validate DataFrame."""
        try:
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Ensure required columns exist and are numeric
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    # Try lowercase version
                    lowercase_col = col.lower()
                    if lowercase_col in df.columns:
                        logger.debug(f"Converting column name from {lowercase_col} to {col}")
                        df = df.rename(columns={lowercase_col: col})
                    else:
                        logger.error(f"Missing required column: {col}")
                        return None

                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Validate numeric conversion
            nan_counts = df[required_columns].isna().sum()
            if nan_counts.any():
                logger.warning(f"NaN values after numeric conversion: {nan_counts}")
                
            # Drop rows with NaN values
            original_len = len(df)
            df = df.dropna(subset=required_columns)
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN values")
            
            # Sort by date
            df = df.sort_index()
            logger.debug(f"Final DataFrame shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}", exc_info=True)
            return None
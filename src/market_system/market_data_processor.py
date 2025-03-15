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

class MarketDataProcessor:
    """
    Processes market system data from JSON files.
    
    Handles validation, loading, and processing of market data files
    with proper error handling and data integrity checks.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the MarketDataProcessor.
        
        Args:
            base_path: Base directory containing market data folders
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise MarketDataError(f"Base path does not exist: {self.base_path}")

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

    def process_market_data(self, market_name: str) -> pd.DataFrame:
        """
        Process all JSON files for a specific market.
        
        Args:
            market_name: Name of the market to process
            
        Returns:
            pd.DataFrame: Processed market data
            
        Raises:
            MarketDataError: If processing fails
        """
        market_path = self.base_path / market_name
        if not market_path.exists():
            raise MarketDataError(f"Market directory not found: {market_path}")

        # Get all JSON files for the market
        json_files = list(market_path.glob("*.json"))
        if not json_files:
            raise MarketDataError(f"No JSON files found in {market_path}")

        # Process each file and combine data
        dfs = []
        for file_path in json_files:
            try:
                logger.info(f"Processing file: {file_path}")
                json_data = self.validate_json_file(file_path)
                
                # Convert JSON data to DataFrame
                df = pd.DataFrame(json_data["data"])
                df["market_name"] = json_data["market_name"]
                df["source_file"] = file_path.name
                
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Handle additional metrics if present
                if "additional_metrics" in df.columns:
                    metrics_df = pd.json_normalize(df["additional_metrics"])
                    df = pd.concat([df.drop("additional_metrics", axis=1), metrics_df], axis=1)
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                raise MarketDataError(f"Failed to process {file_path}: {str(e)}")

        # Combine all DataFrames
        if not dfs:
            raise MarketDataError(f"No valid data found for market: {market_name}")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp and remove duplicates
        combined_df = (combined_df
                      .sort_values("timestamp")
                      .drop_duplicates(subset=["timestamp", "market_name"], keep="last")
                      .reset_index(drop=True))
        
        logger.info(f"Successfully processed {len(json_files)} files for {market_name}")
        return combined_df

    def get_market_summary(self, market_name: str) -> Dict:
        """
        Generate summary statistics for a market's data.
        
        Args:
            market_name: Name of the market to summarize
            
        Returns:
            Dict: Summary statistics
        """
        try:
            df = self.process_market_data(market_name)
            
            summary = {
                "market_name": market_name,
                "data_points": len(df),
                "date_range": {
                    "start": df["timestamp"].min().isoformat(),
                    "end": df["timestamp"].max().isoformat()
                },
                "price_metrics": {
                    "mean": df["price"].mean(),
                    "median": df["price"].median(),
                    "min": df["price"].min(),
                    "max": df["price"].max()
                },
                "volume_metrics": {
                    "total": df["volume"].sum(),
                    "mean": df["volume"].mean(),
                    "median": df["volume"].median()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for {market_name}: {str(e)}")
            raise MarketDataError(f"Failed to generate summary for {market_name}: {str(e)}")
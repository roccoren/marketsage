#!/usr/bin/env python
"""
Transform stock symbols from CSV files into structured JSON format.

This script reads CSV files containing stock listings and transforms them
into structured JSON files suitable for use with the market data fetcher.
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockListingTransformer:
    """Transforms stock listing CSV files into structured JSON format."""

    def __init__(
        self,
        input_dir: str = "config/orignal",
        output_dir: str = "config/market"
    ) -> None:
        """
        Initialize the transformer with input and output directories.
        
        Args:
            input_dir: Directory containing the original CSV files
            output_dir: Directory to save the transformed JSON files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Market-specific configurations
        self.market_configs = {
            "nasdaq": {
                "input_file": "nasdaq-listed.csv",
                "output_file": "nasdaq-listed.json",
                "symbol_field": "Symbol",
                "name_field": "Security Name",
                "market_name": "NASDAQ",
                "description": "NASDAQ Listed Companies",
                "exchange_info": {
                    "mic": "XNAS",
                    "website": "https://www.nasdaq.com"
                },
                "market_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "America/New_York"
                }
            },
            "nyse": {
                "input_file": "nyse-listed.csv",
                "output_file": "nyse-listed.json",
                "symbol_field": "ACT Symbol",
                "name_field": "Company Name",
                "market_name": "NYSE",
                "description": "New York Stock Exchange Listed Companies",
                "exchange_info": {
                    "mic": "XNYS",
                    "website": "https://www.nyse.com"
                },
                "market_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "America/New_York"
                }
            },
        }
    
    def read_csv_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Read a CSV file and return its contents as a list of dictionaries.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries representing the CSV rows
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            csv.Error: For CSV parsing errors
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                # Use DictReader to automatically use the header row as field names
                reader = csv.DictReader(csvfile)
                return list(reader)
        except csv.Error as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise
    
    def transform_listings(self, market: str) -> Dict[str, Any]:
        """
        Transform stock listings for a specific market to the new format with
        symbols array and metadata.
        
        Args:
            market: Market identifier (e.g., 'nasdaq', 'nyse')
            
        Returns:
            Dictionary with symbols array and metadata
            
        Raises:
            ValueError: If the market is not supported
            Various exceptions from read_csv_file
        """
        if market not in self.market_configs:
            raise ValueError(f"Unsupported market: {market}")
        
        config = self.market_configs[market]
        input_file = self.input_dir / config["input_file"]
        
        try:
            # Read and parse the CSV file
            rows = self.read_csv_file(input_file)
            
            # Get field names from config
            symbol_field = config["symbol_field"]
            name_field = config["name_field"]
            
            # Validate CSV structure
            if rows and (symbol_field not in rows[0] or name_field not in rows[0]):
                missing_fields = []
                if symbol_field not in rows[0]:
                    missing_fields.append(symbol_field)
                if name_field not in rows[0]:
                    missing_fields.append(name_field)
                raise ValueError(
                    f"CSV missing required fields: {', '.join(missing_fields)}"
                )
            
            # Extract symbols and create symbol details dictionary
            symbols = []
            symbol_details = {}
            
            for row in rows:
                symbol = row[symbol_field].strip()
                name = row[name_field].strip()
                
                # Skip empty entries
                if not symbol:
                    continue
                
                # Add to symbols list
                symbols.append(symbol)
                
                # Add to symbol details dictionary
                symbol_details[symbol] = {
                    "name": name,
                    "symbol": symbol
                }
            
            # Create metadata structure
            current_date = datetime.now().strftime("%Y-%m-%d")
            metadata = {
                "market": config["market_name"],
                "description": config["description"],
                "last_updated": current_date,
                "total_symbols": len(symbols),
                "market_hours": config["market_hours"],
                "exchange_info": config["exchange_info"]
            }
            
            # Create the final result structure
            result = {
                "symbols": symbols,
                "metadata": metadata,
                "symbol_details": symbol_details
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error transforming listings for {market}: {str(e)}")
            raise
    
    def save_json_file(
        self, 
        data: Dict[str, Any], 
        file_path: Path,
        indent: int = 2
    ) -> None:
        """
        Save data as a JSON file.
        
        Args:
            data: Data to be saved
            file_path: Path where the JSON file will be saved
            indent: Indentation level for the JSON file (pretty-printing)
            
        Raises:
            IOError: If there's an error writing the file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=indent)
            logger.info(f"Saved JSON file: {file_path}")
        except IOError as e:
            logger.error(f"Error saving JSON file {file_path}: {str(e)}")
            raise
    
    def process_market(self, market: str) -> Tuple[bool, str]:
        """
        Process a single market: transform CSV to JSON and save the result.
        
        Args:
            market: Market identifier (e.g., 'nasdaq', 'nyse')
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Processing {market} listings...")
            data = self.transform_listings(market)
            
            # Get output file path from config
            output_file = self.output_dir / self.market_configs[market]["output_file"]
            
            # Save transformed data
            self.save_json_file(data, output_file)
            
            return True, f"Successfully processed {market} listings: {len(data['symbols'])} symbols"
        
        except Exception as e:
            return False, f"Failed to process {market} listings: {str(e)}"
    
    def process_all_markets(self) -> bool:
        """
        Process all configured markets.
        
        Returns:
            Boolean indicating overall success
        """
        all_successful = True
        
        for market in self.market_configs:
            success, message = self.process_market(market)
            logger.info(message)
            
            if not success:
                all_successful = False
        
        return all_successful


def main():
    """CLI entry point for stock listing transformer."""
    parser = argparse.ArgumentParser(
        description="Transform stock symbols from CSV files to JSON format"
    )
    
    # Directories
    parser.add_argument(
        '--input-dir',
        type=str,
        default="config/orignal",
        help='Directory containing the original CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="config/market",
        help='Directory to save the transformed JSON files'
    )
    
    # Market selection
    parser.add_argument(
        '-m', '--market',
        type=str,
        choices=['nasdaq', 'nyse', 'all'],
        default='all',
        help='Specific market to process or "all" for all markets'
    )
    
    args = parser.parse_args()
    
    # Initialize transformer
    transformer = StockListingTransformer(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process requested market(s)
    if args.market == 'all':
        success = transformer.process_all_markets()
    else:
        success, message = transformer.process_market(args.market)
        logger.info(message)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
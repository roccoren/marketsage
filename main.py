#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import asyncio
from pathlib import Path
from typing import Dict

class ScannerConfig:
    def __init__(self, config_data: Dict):
        # Initialize configuration settings
        self.exchanges = config_data.get('exchanges', [])
        self.providers = config_data.get('providers', {})
        self.data_directory = config_data.get('dataDirectory', 'data')
        self.output_directory = config_data.get('outputDirectory', 'output')
        self.log_directory = config_data.get('logDirectory', 'logs')

    def validate(self) -> None:
        if not self.exchanges:
            raise ValueError("No exchanges configured")
        
        if not self.data_directory:
            raise ValueError("Data directory not specified")
        
        if not self.output_directory:
            raise ValueError("Output directory not specified")
        
        if not self.log_directory:
            raise ValueError("Log directory not specified")
        
        yahoo_finance = self.providers.get('yahooFinance', {})
        if not yahoo_finance.get('rateLimit'):
            raise ValueError("Yahoo Finance rate limit not specified")
        
        if not yahoo_finance.get('batchSize'):
            raise ValueError("Yahoo Finance batch size not specified")
        
        # Validate LongPort if configured
        long_port = self.providers.get('longPort', {})
        if long_port.get('apiKey'):
            if not long_port.get('apiSecret'):
                raise ValueError("LongPort API secret not specified")
            if not long_port.get('rateLimit'):
                raise ValueError("LongPort rate limit not specified")
            if not long_port.get('batchSize'):
                raise ValueError("LongPort batch size not specified")

def load_config(config_path: str) -> ScannerConfig:
    """Load and validate configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = ScannerConfig(config_data)
        config.validate()
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

def update_config_for_exchange(config: ScannerConfig, market: str) -> None:
    """Update configuration to only include the selected market."""
    config.exchanges = [exchange for exchange in config.exchanges 
                       if exchange['name'] == market]

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Market Scanner')
    parser.add_argument('--config', default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--market', required=True,
                      help='Market to scan (ASX, HKEX, NASDAQ, NYSE)')
    parser.add_argument('--update', action='store_true',
                      help='Market to scan (ASX, HKEX, NASDAQ, NYSE)')
    args = parser.parse_args()

    try:
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Create a file handler for logging
        file_handler = logging.FileHandler('logs/scanner.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger = logging.getLogger('StockScanner')
        logger.addHandler(file_handler)

        logger.info('Starting the main function execution...')  # Added logging statement
        print('Starting the main function execution...')  # Added print statement

        # Validate market argument
        from scanner import StockScanner
        market = args.market.upper()
        scanner = StockScanner(None)  # Temporary instance for validation
        if not scanner.validate_exchange(market):
            logger.error(f"Error: Invalid market '{market}'. Valid markets are: {', '.join(scanner.VALID_EXCHANGES)}")
            sys.exit(1)
        del scanner  # Clean up temporary instance

        # Load configuration
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")

        # Create necessary directories
        Path(config.data_directory).mkdir(exist_ok=True)
        Path(config.output_directory).mkdir(exist_ok=True)
        Path(config.log_directory).mkdir(exist_ok=True)

        # Update config to only include selected market
        update_config_for_exchange(config, market)
        
        # Initialize scanner with config and strategy
        from strategies import PriceCrossoverStrategy
        scanner = StockScanner(config)
        crossover_strategy = PriceCrossoverStrategy()
        scanner.register_strategy(crossover_strategy)

        logger.info('Scanner initialized with configuration and strategy.')  # Added logging statement

        # Update stock data if requested
        if args.update:
            logger.info('Stock data update requested.')  # Added logging statement
            await scanner.update_stock_data()
            logger.info("Stock data update completed")

        # Run scanner
        logger.info(f"Starting market scan for {market}...")
        await scanner.scan()
        logger.info("Scan completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
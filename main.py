import argparse

parser = argparse.ArgumentParser(description='Stock Scanner')
parser.add_argument('-exchanges', type=str, nargs='+', help='List of exchanges to scan, e.g., HKEX')
args = parser.parse_args()

import asyncio
import logging
from pathlib import Path
import json
from typing import List, Dict, Any

from scanner import StockScanner
from strategies.strategy_rsi_stochastic import RSIStochStrategy
from strategies.strategy_atr_moving_average import ATRMovingAverageStrategy
from strategies.strategy_adx_macd_cci import ADXMACDCCI_Strategy
from strategies.strategy_obv import OBVStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Configuration:
    def __init__(self, config_data: Dict[str, Any]):
        self.data_directory = config_data.get('data_directory', 'data')
        self.output_directory = config_data.get('output_directory', 'output')
        self.exchanges = config_data.get('exchanges', [])
        self.providers = config_data.get('providers', {})

async def main():
    try:
        # Load configuration
        with open('config.json', 'r') as f:
            config_data = json.load(f)
        
        # Override exchanges if provided via command line
        if args.exchanges:
            config_data['exchanges'] = [{'name': exchange} for exchange in args.exchanges]
        
        config = Configuration(config_data)
        
        # Initialize scanner
        scanner = StockScanner(config)
        
        # Register all strategies
        strategies = [
            RSIStochStrategy(),
            ATRMovingAverageStrategy(),
            ADXMACDCCI_Strategy(),
            OBVStrategy()
        ]
        
        for strategy in strategies:
            scanner.register_strategy(strategy)
        
        # Run the scan with all registered strategies
        logger.info("Running scan with all strategies...")
        await scanner.scan(days=60)  # Use last 60 days of data for analysis
        
        logger.info("Scan completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scan: {str(e)}")
        raise

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Run the async main function
    asyncio.run(main())

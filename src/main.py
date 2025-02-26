import argparse

parser = argparse.ArgumentParser(description='Stock Scanner')
parser.add_argument('-exchanges', type=str, nargs='+', help='List of exchanges to scan, e.g., HKEX')
parser.add_argument('--strategies', type=str, help='Comma-separated list of strategies to use (rsi_stoch,atr_ma,adx_macd_cci,obv)')
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
from utils.report_display import load_latest_report, display_analysis_report

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
        # Get project root directory (parent of src)
        root_dir = Path(__file__).parent.parent
        
        # Load configuration
        with open(root_dir / "config" / "config.json", 'r') as f:
            config_data = json.load(f)
        
        # Override exchanges if provided via command line
        if args.exchanges:
            config_data['exchanges'] = [{'name': exchange} for exchange in args.exchanges]
        
        config = Configuration(config_data)
        
        # Initialize scanner
        scanner = StockScanner(config)
        
        # Available strategies mapping
        strategy_map = {
            'rsi_stoch': RSIStochStrategy,
            'atr_ma': ATRMovingAverageStrategy,
            'adx_macd_cci': ADXMACDCCI_Strategy,
            'obv': OBVStrategy
        }
        
        # Initialize selected strategies or all if none specified
        strategies = []
        if args.strategies:
            selected = args.strategies.split(',')
            for strategy_name in selected:
                if strategy_name in strategy_map:
                    strategies.append(strategy_map[strategy_name]())
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
        else:
            strategies = [strategy() for strategy in strategy_map.values()]
            
        if not strategies:
            raise ValueError(f"No valid strategies selected. Available strategies: {', '.join(strategy_map.keys())}")
        
        for strategy in strategies:
            scanner.register_strategy(strategy)
        
        # Run the scan with all registered strategies
        logger.info("Running scan with all strategies...")
        await scanner.scan(days=60)  # Use last 60 days of data for analysis
        
        logger.info("Scan completed successfully")
        
        # Display the analysis report
        try:
            latest_report = load_latest_report(config.output_directory)
            report_text = display_analysis_report(latest_report)
            print("\nTRADING ANALYSIS REPORT")
            print("=" * 80)
            print(report_text)
        except FileNotFoundError as e:
            logger.error(f"Could not display trading analysis report: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing trading analysis report: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error during scan: {str(e)}")
        raise

async def async_main():
    """Main async function for the stock scanner"""
    # Get project root directory (parent of src)
    root_dir = Path(__file__).parent.parent
    
    # Create necessary directories if they don't exist
    Path(root_dir / "data").mkdir(exist_ok=True)
    Path(root_dir / "output").mkdir(exist_ok=True)
    
    try:
        # Load configuration
        with open(root_dir / "config" / "config.json", 'r') as f:
            config_data = json.load(f)
        
        # Rest of the original async main function
        if args.exchanges:
            config_data['exchanges'] = [{'name': exchange} for exchange in args.exchanges]
        
        config = Configuration(config_data)
        scanner = StockScanner(config)
        
        strategy_map = {
            'rsi_stoch': RSIStochStrategy,
            'atr_ma': ATRMovingAverageStrategy,
            'adx_macd_cci': ADXMACDCCI_Strategy,
            'obv': OBVStrategy
        }
        
        strategies = []
        if args.strategies:
            selected = args.strategies.split(',')
            for strategy_name in selected:
                if strategy_name in strategy_map:
                    strategies.append(strategy_map[strategy_name]())
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
        else:
            strategies = [strategy() for strategy in strategy_map.values()]
            
        if not strategies:
            raise ValueError(f"No valid strategies selected. Available strategies: {', '.join(strategy_map.keys())}")
        
        for strategy in strategies:
            scanner.register_strategy(strategy)
        
        logger.info("Running scan with all strategies...")
        await scanner.scan(days=60)
        
        logger.info("Scan completed successfully")
        
        try:
            latest_report = load_latest_report(config.output_directory)
            report_text = display_analysis_report(latest_report)
            print("\nTRADING ANALYSIS REPORT")
            print("=" * 80)
            print(report_text)
        except FileNotFoundError as e:
            logger.error(f"Could not display trading analysis report: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing trading analysis report: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during scan: {str(e)}")
        raise

def main():
    """Entry point for the stock scanner"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

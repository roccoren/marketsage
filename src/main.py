import argparse

parser = argparse.ArgumentParser(description='Stock Scanner')
parser.add_argument('--exchanges', type=str, nargs='+', help='List of exchanges to scan, e.g., HKEX')
parser.add_argument('--strategies', type=str, help='Comma-separated list of strategies to use (rsi_stoch,atr_ma,adx_macd_cci,obv)')
args = parser.parse_args()

import asyncio
import logging
from pathlib import Path
import json
from typing import List, Dict, Any

from src.scanner import StockScanner
from src.strategies.strategy_rsi_stochastic import RSIStochStrategy
from src.strategies.strategy_atr_moving_average import ATRMovingAverageStrategy
from src.strategies.strategy_adx_macd_cci import ADXMACDCCI_Strategy
from src.strategies.strategy_obv import OBVStrategy
from src.utils.report_display import load_latest_report, display_analysis_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Configuration:
    def __init__(self, config_data: Dict[str, Any]):
        self.data_directory = config_data.get('data_directory', 'data')
        self.output_directory = config_data.get('output_directory', 'output')
        self.exchanges = config_data.get('exchanges', [])
        self.providers = config_data.get('providers', {})

def initialize_strategies() -> List[Any]:
    """Initialize and return list of trading strategies based on command line arguments."""
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
    
    return strategies

async def display_report(output_directory: str) -> None:
    """Display the analysis report from the latest scan."""
    try:
        latest_report = load_latest_report(output_directory)
        if latest_report is None:
            logger.warning("No valid trading analysis report found")
            return
            
        report_text = display_analysis_report(latest_report)
        print("\nTRADING ANALYSIS REPORT")
        print("=" * 80)
        print(report_text)
    except FileNotFoundError as e:
        logger.error(f"Could not display trading analysis report: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing trading analysis report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

async def main() -> None:
    """Main async function for the stock scanner"""
    try:
        root_dir = Path(__file__).parent.parent
        
        # Create necessary directories
        Path(root_dir / "data").mkdir(exist_ok=True)
        Path(root_dir / "output").mkdir(exist_ok=True)
        
        # Load and process configuration
        with open(root_dir / "config" / "config.json", 'r') as f:
            config_data = json.load(f)
        
        if args.exchanges:
            config_data['exchanges'] = [{'name': exchange} for exchange in args.exchanges]
        
        config = Configuration(config_data)
        scanner = StockScanner(config)
        
        # Initialize and register strategies
        strategies = initialize_strategies()
        for strategy in strategies:
            scanner.register_strategy(strategy)
        
        # Run scan
        logger.info("Running scan with all strategies...")
        await scanner.scan(days=60)
        logger.info("Scan completed successfully")
        
        # Display results
        await display_report(config.output_directory)
        
    except Exception as e:
        logger.error(f"Error during scan: {str(e)}")
        raise

def run():
    """Entry point for the stock scanner"""
    asyncio.run(main())

if __name__ == "__main__":
    run()

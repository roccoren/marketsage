#!/usr/bin/env python3
import os
import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

def get_latest_params_file(params_dir: Path) -> Path:
    """
    Get the most recent parameters file from the optimization output directory
    
    Args:
        params_dir: Directory containing optimization parameter files
        
    Returns:
        Path to the latest parameters file
    """
    if not params_dir.exists():
        logger.error(f"Optimization directory {params_dir} does not exist")
        return None
        
    param_files = list(params_dir.glob('optimal_params_*.json'))
    
    if not param_files:
        logger.error(f"No parameter files found in {params_dir}")
        return None
        
    # Sort by modification time (newest first)
    latest_file = max(param_files, key=os.path.getmtime)
    logger.info(f"Using parameter file: {latest_file}")
    
    return latest_file

def load_parameters(param_file: Path) -> dict:
    """
    Load strategy parameters from a JSON file
    
    Args:
        param_file: Path to parameters file
        
    Returns:
        Dictionary of parameters
    """
    if not param_file or not param_file.exists():
        logger.error("Parameter file does not exist")
        return {}
        
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        logger.info(f"Loaded parameters from {param_file}")
        return params
    
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        return {}

def run_backtest_with_params(params: dict, symbols: list = None, start_date: str = '2020-01-01'):
    """
    Run the NYSE backtest system with the provided parameters
    
    Args:
        params: Strategy parameters
        symbols: List of symbols to backtest (None for all)
        start_date: Start date for backtesting
    """
    from nyse_backtest_system import main as run_backtest
    
    # Store parameters in a temporary file
    params_file = Path('temp_params.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Set environment variables for the backtest
    os.environ['BACKTEST_PARAMS_FILE'] = str(params_file)
    
    if symbols:
        os.environ['BACKTEST_SYMBOLS'] = ','.join(symbols)
    
    os.environ['BACKTEST_START_DATE'] = start_date
    
    # Run the backtest
    run_backtest()
    
    # Clean up
    if params_file.exists():
        params_file.unlink()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run NYSE backtesting system with optimized parameters'
    )
    
    parser.add_argument(
        '--params', 
        type=str,
        help='Path to parameters JSON file'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to backtest'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--optimize-first',
        action='store_true',
        help='Run optimization before backtesting'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # If optimization is requested, run it first
        if args.optimize_first:
            logger.info("Running strategy optimization first...")
            from optimize_strategy import main as run_optimization
            run_optimization()
            logger.info("Optimization complete, proceeding to backtest...")
        
        # Load parameters
        params = {}
        if args.params:
            params = load_parameters(Path(args.params))
        else:
            # Look for the latest parameter file from optimization
            params_dir = Path('output/optimization')
            latest_params_file = get_latest_params_file(params_dir)
            
            if latest_params_file:
                params = load_parameters(latest_params_file)
        
        if not params:
            logger.error("No parameters available. Please provide parameters or run optimization first.")
            return
            
        # Parse symbols if provided
        symbols = None
        if args.symbols:
            symbols = args.symbols.split(',')
            logger.info(f"Running backtest for symbols: {symbols}")
        
        # Run the backtest with the parameters
        run_backtest_with_params(params, symbols, args.start_date)
        
    except Exception as e:
        logger.exception("An error occurred during execution")
        raise

if __name__ == "__main__":
    main()
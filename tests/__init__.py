"""Test package initialization and configuration."""

import sys
from pathlib import Path

# Add project root to Python path to allow imports from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'chunk_size': 1000,  # Smaller chunk size for testing
    'max_workers': 2,    # Limited workers for testing
    'test_market': 'test_market',
    'test_symbol': 'TEST.SYM',
    'date_format': '%Y-%m-%d'
}

def create_test_data(start_date: str, num_days: int) -> dict:
    """
    Create sample market data for testing.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        num_days: Number of days of data to generate
        
    Returns:
        dict: Sample market data
    """
    from datetime import datetime, timedelta
    import numpy as np
    
    base_date = datetime.strptime(start_date, TEST_CONFIG['date_format'])
    base_price = 100.0
    
    return {
        'date': base_date.strftime(TEST_CONFIG['date_format']),
        'symbol': TEST_CONFIG['test_symbol'],
        'market': TEST_CONFIG['test_market'],
        'open': base_price * (1 + np.random.normal(0, 0.01)),
        'high': base_price * (1 + np.random.normal(0, 0.02)),
        'low': base_price * (1 - np.random.normal(0, 0.02)),
        'close': base_price * (1 + np.random.normal(0, 0.01)),
        'volume': int(np.random.randint(1000, 10000)),
        'trading_status': 'active',
        'currency': 'USD',
        'metrics': {
            'indicators': {
                'vwap': base_price * (1 + np.random.normal(0, 0.005)),
                'turnover': int(base_price * np.random.randint(1000, 10000))
            },
            'returns': {
                'daily': np.random.normal(0, 0.01),
                'intraday_range': np.random.uniform(0.01, 0.03)
            }
        }
    }

def setup_test_data_directory() -> Path:
    """
    Create and set up a temporary test data directory.
    
    Returns:
        Path: Path to test directory
    """
    import tempfile
    test_dir = Path(tempfile.mkdtemp())
    market_dir = test_dir / TEST_CONFIG['test_market']
    symbol_dir = market_dir / TEST_CONFIG['test_symbol']
    symbol_dir.mkdir(parents=True)
    return test_dir

def cleanup_test_data(test_dir: Path):
    """
    Clean up test data directory.
    
    Args:
        test_dir: Path to test directory to clean up
    """
    import shutil
    shutil.rmtree(test_dir)
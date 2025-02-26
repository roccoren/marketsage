#!/usr/bin/env python3
"""
Performance test script comparing optimized vs unoptimized yfinance data fetching.
"""
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import asyncio
import time
import logging
import yfinance as yf
from datetime import datetime, timedelta
from src.utils.async_utils import YahooFinanceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data
TEST_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
    'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    '0700.HK', '9988.HK', '3690.HK', '9618.HK', '2318.HK'
]

async def test_optimized_fetch():
    """Test optimized data fetching using YahooFinanceClient."""
    logger.info("Testing optimized fetching...")
    start_time = time.time()
    
    # Use optimized client
    async with YahooFinanceClient(data_dir="test_data") as client:
        # Get historical data for all symbols
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = await client.get_history_batch(
            TEST_SYMBOLS,
            start_date=start_date,
            end_date=end_date
        )
        
        success_count = sum(1 for data in results.values() if data is not None)
    
    duration = time.time() - start_time
    return duration, success_count

def test_basic_fetch():
    """Test basic yfinance data fetching."""
    logger.info("Testing basic fetching...")
    start_time = time.time()
    success_count = 0
    
    # Basic yfinance download
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    for symbol in TEST_SYMBOLS:
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            if not data.empty:
                success_count += 1
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    
    duration = time.time() - start_time
    return duration, success_count

async def run_performance_test():
    """Run performance comparison test."""
    # Test basic fetching
    basic_duration, basic_success = test_basic_fetch()
    logger.info(f"Basic fetching: {basic_duration:.2f}s, {basic_success}/{len(TEST_SYMBOLS)} successful")
    
    # Test optimized fetching
    opt_duration, opt_success = await test_optimized_fetch()
    logger.info(f"Optimized fetching: {opt_duration:.2f}s, {opt_success}/{len(TEST_SYMBOLS)} successful")
    
    # Calculate improvement
    time_improvement = ((basic_duration - opt_duration) / basic_duration) * 100
    success_improvement = ((opt_success - basic_success) / len(TEST_SYMBOLS)) * 100
    
    logger.info("\nPerformance Summary:")
    logger.info(f"Time improvement: {time_improvement:.1f}%")
    logger.info(f"Success rate improvement: {success_improvement:.1f}%")

    # Optional memory usage tracking
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Final memory usage: {memory_mb:.1f} MB")
    except ImportError:
        logger.info("\nNote: Memory usage tracking requires psutil package")
        logger.info("Install with: pip install psutil")
    except Exception as e:
        logger.warning(f"Could not measure memory usage: {e}")

if __name__ == "__main__":
    # Create test data directory
    os.makedirs("test_data", exist_ok=True)

    # Verify we're in the project root
    if not os.path.exists('src') or not os.path.exists('setup.py'):
        print("Error: Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        print("Required structure:")
        print("  - src/")
        print("  - setup.py")
        sys.exit(1)
    
    # Run tests
    asyncio.run(run_performance_test())
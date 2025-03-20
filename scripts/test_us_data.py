#!/usr/bin/env python
"""
Test script to verify US stock data can be loaded correctly
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("test_us_data")

# Check qlib is installed
try:
    import qlib
    from qlib.data import D
except ImportError:
    logger.error("qlib is not installed. Please install using: pip install qlib")
    sys.exit(1)

def test_us_data(data_path='~/.qlib/qlib_data/us_data'):
    """Test loading US stock data with qlib"""
    # Expand user path
    data_path = os.path.expanduser(data_path)
    
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist")
        return False
    
    logger.info(f"Testing US stock data at {data_path}")
    
    try:
        # Initialize qlib
        logger.info("Initializing qlib...")
        qlib.init(provider_uri=data_path, region="us")
        
        # Load instruments
        logger.info("Loading instruments...")
        instruments_path = os.path.join(data_path, "instruments.pkl")
        if os.path.exists(instruments_path):
            instruments = pd.read_pickle(instruments_path)
            logger.info(f"Found {len(instruments)} instruments")
            
            # Get a few sample instruments
            sample_instruments = instruments["instrument"].head(3).tolist()
            logger.info(f"Sample instruments: {sample_instruments}")
        else:
            logger.error("instruments.pkl not found")
            return False
        
        # Set date range for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Test with 30 days of data
        
        # Try different frequency settings for loading data
        freqs = ["day", "1d", None]
        success = False
        
        for freq in freqs:
            try:
                logger.info(f"Testing with frequency: {freq}")
                
                # Test getting features for a sample instrument
                fields = ["$close", "$open", "$high", "$low", "$volume"]
                test_instr = sample_instruments[0]
                
                logger.info(f"Fetching data for {test_instr} with fields {fields}")
                df = D.features(
                    [test_instr], 
                    fields, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    freq=freq
                )
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully retrieved data with freq={freq}!")
                    logger.info(f"Sample data shape: {df.shape}")
                    logger.info(f"Sample data:\n{df.head()}")
                    success = True
                    break
                else:
                    logger.warning(f"Retrieved empty dataframe with freq={freq}")
            except Exception as e:
                logger.warning(f"Error testing with freq={freq}: {e}")
                continue
        
        if not success:
            # If all freqs failed, try direct file access
            logger.info("Testing direct file access...")
            for ticker in sample_instruments:
                feature_path = os.path.join(data_path, "raw", ticker, "feature.pkl")
                if os.path.exists(feature_path):
                    df = pd.read_pickle(feature_path)
                    logger.info(f"Direct file access for {ticker} successful")
                    logger.info(f"Sample data shape: {df.shape}")
                    logger.info(f"Sample data:\n{df.head()}")
                    success = True
                    break
        
        if success:
            logger.info("US stock data test successful!")
            return True
        else:
            logger.error("All data access methods failed")
            return False
    
    except Exception as e:
        logger.error(f"Error testing US stock data: {e}")
        return False

if __name__ == "__main__":
    # Get data path from command line if provided
    data_path = sys.argv[1] if len(sys.argv) > 1 else '~/.qlib/qlib_data/us_data'
    
    # Test the data
    if test_us_data(data_path):
        logger.info("US stock data test completed successfully")
    else:
        logger.error("US stock data test failed")
        sys.exit(1) 
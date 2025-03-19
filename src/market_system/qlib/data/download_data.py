"""
Script to download US stock market data using Qlib.
"""

import qlib
from qlib.data import D
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config, get_module_by_module_path
import pandas as pd
from pathlib import Path
from ..config import QLIB_CONFIG

def download_us_data(start_date: str = "2019-01-01", 
                    end_date: str = None,
                    data_dir: str = None):
    """Download US stock market data.
    
    Args:
        start_date (str): Start date for data download (YYYY-MM-DD)
        end_date (str, optional): End date for data download (YYYY-MM-DD)
        data_dir (str, optional): Directory to store the data
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "yahoo_data")
    
    # Initialize Qlib with US stock market
    qlib.init(
        provider_uri=data_dir,
        region=REG_US,
        expression_cache=None,
        calendar=None
    )

    # Configure data provider
    provider_config = {
        "class": "YahooFinanceNormalize",
        "module_path": "qlib.data.data_loader.yahoo",
        "kwargs": {
            "start": start_date,
            "end": end_date,
            "interval": "1d",
            "region": "US",
            "symbols": None,  # Download all available symbols
            "show_progress": True,
        },
    }

    print(f"Downloading US stock data from {start_date}...")
    provider = init_instance_by_config(provider_config)
    provider.dump_data()
    print("Data download completed!")

def normalize_data():
    """Normalize the downloaded data according to Qlib format."""
    print("Normalizing data...")
    
    # Get the normalize class
    normalize_class = get_module_by_module_path("qlib.data.data_loader.yahoo.YahooFinanceNormalize")
    
    # Run normalization
    normalizer = normalize_class()
    normalizer.normalize()
    print("Data normalization completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download US stock market data")
    parser.add_argument("--start_date", type=str, default="2019-01-01",
                      help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None,
                      help="End date (YYYY-MM-DD)")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="Directory to store the data")
    
    args = parser.parse_args()
    
    # Download and normalize data
    download_us_data(args.start_date, args.end_date, args.data_dir)
    normalize_data() 
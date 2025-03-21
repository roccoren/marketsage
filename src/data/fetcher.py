import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Union, Tuple
from src.utils.init import init_mlflow, close_mlflow
from src.config import MLFLOW_CONFIG 
from qlib.constant import REG_US, REG_CN
from qlib.utils import exists_qlib_data


def fetch_stock_data(ticker, start_date, end_date, interval="1d"):
    """Fetch stock data for a specific ticker"""
    try:
        # Download data from yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        
        # If data is empty, return None
        if data.empty:
            print(f"No data available for {ticker} from {start_date} to {end_date}")
            return None, ticker
            
        # Rename columns to match Qlib format
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close', 
            'Volume': 'volume'
        })
        
        # Drop unnecessary columns
        data = data[['open', 'high', 'low', 'close', 'volume']]
        
        # Reset index to have Date as a column
        data = data.reset_index()
        
        # Convert Date to string in YYYY-MM-DD format
        data['date'] = data['Date'].dt.strftime('%Y-%m-%d')
        data = data.drop('Date', axis=1)
        
        # Reorder columns to match Qlib format
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"Successfully fetched data for {ticker}")
        return data, ticker
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None, ticker


def get_market_symbols(market, region):
    """Get list of stock symbols for a specific market"""
    # For demonstration purposes, using small sample sets
    # In a real implementation, you'd fetch the actual constituents of these indices
    
    if region.lower() == "us":
        if market.lower() == "sp500":
            # Sample of SP500 stocks (in production, fetch the full list)
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "PG", "UNH"]
        elif market.lower() == "nasdaq":
            # Sample of NASDAQ stocks
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ADBE", "NFLX", "INTC"]
        else:
            # Default to a few major US stocks
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    else:  # Chinese market
        # For Chinese stocks, append .SS for Shanghai and .SZ for Shenzhen
        if market.lower() == "csi300":
            # Sample of CSI300 stocks (in production, fetch the full list)
            return ["600519.SS", "601318.SS", "600036.SS", "601398.SS", "601988.SS", 
                   "000333.SZ", "000651.SZ", "000858.SZ", "002415.SZ", "300059.SZ"]
        else:
            # Default to a few major Chinese stocks
            return ["600519.SS", "601318.SS", "600036.SS", "000333.SZ", "000651.SZ"]


def fetch_and_save_to_qlib(stock_symbols=None, start_date=None, end_date=None, interval="1d", qlib_dir=None, region=None, max_workers=16, market=None):
    """Fetch stock data and save in Qlib format"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if stock_symbols is None:
        if market:
            stock_symbols = get_market_symbols(market, region or "us")
        else:
            # Default to a few major stocks based on region
            stock_symbols = get_market_symbols("default", region or "us")
    
    # Make sure qlib_dir exists
    os.makedirs(qlib_dir, exist_ok=True)
    
    # Create 'calendars' and 'instruments' directories
    calendars_dir = os.path.join(qlib_dir, 'calendars')
    instruments_dir = os.path.join(qlib_dir, 'instruments')
    features_dir = os.path.join(qlib_dir, 'features')
    
    os.makedirs(calendars_dir, exist_ok=True)
    os.makedirs(instruments_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Create features directories
    for feature in ['open', 'high', 'low', 'close', 'volume']:
        os.makedirs(os.path.join(features_dir, feature), exist_ok=True)
    
    # List to store all unique dates (for calendar)
    all_dates = set()
    
    # Dictionary to map tickers to their data
    ticker_data = {}
    
    # Fetch data for all tickers in parallel
    print(f"Fetching data for {len(stock_symbols)} symbols from {start_date} to {end_date}")
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data, ticker, start_date, end_date, interval): ticker for ticker in stock_symbols}
        for future in futures:
            data, ticker = future.result()
            if data is not None:
                results.append((data, ticker))
    
    # Process results
    success_count = 0
    for data, ticker in results:
        if data is not None:
            ticker_data[ticker] = data
            # Add dates to the set of all dates
            all_dates.update(data['date'].unique())
            success_count += 1
    
    print(f"Successfully fetched data for {success_count} out of {len(stock_symbols)} symbols")
    
    # Save individual stock data files (instruments)
    instruments_file = os.path.join(instruments_dir, 'all.txt')
    with open(instruments_file, 'w') as f:
        for ticker in ticker_data:
            # Write each ticker to the instruments file
            clean_ticker = ticker.replace('.', '_')
            f.write(f"{ticker}\n")
            
            # Save feature data for each ticker
            for feature in ['open', 'high', 'low', 'close', 'volume']:
                feature_data = ticker_data[ticker][['date', feature]]
                feature_file = os.path.join(features_dir, feature, f"{clean_ticker}.{feature}")
                feature_data.to_csv(feature_file, index=False, header=False, sep=',')
    
    # Save calendar file
    calendar_file = os.path.join(calendars_dir, 'all.txt')
    with open(calendar_file, 'w') as f:
        for date in sorted(all_dates):
            f.write(f"{date}\n")
    
    print(f"Data saved in Qlib format to {qlib_dir}")
    return ticker_data


def default_dump_csvs_from_yahoo(stock_list=None, market_type="all", qlib_dir=None, start_date="2015-01-01", end_date=None, max_workers=16):
    """Download data from Yahoo Finance and convert to Qlib format."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    qlib_dir = qlib_dir or os.path.join(os.getcwd(), "data", "qlib_cn")
    print(f"Using default_dump_csvs_from_yahoo for market {market_type} from {start_date} to {end_date}")
    
    # Use the real implementation to fetch and save data
    fetch_and_save_to_qlib(
        stock_symbols=stock_list,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        qlib_dir=qlib_dir,
        region="cn",
        max_workers=max_workers,
        market=market_type
    )


class YahooDataFetcher:
    def __init__(self, region="us"):
        self.region = region
    
    def fetch_data(self, start_date=None, end_date=None, symbols_file=None, track_mlflow=True):
        """Fetch Yahoo Finance data"""
        print(f"Fetching Yahoo Finance data for region {self.region}")
        
        # Get symbols from file if provided
        symbols = None
        if symbols_file and os.path.exists(symbols_file):
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(symbols)} symbols from {symbols_file}")
        
        # Setup qlib directory
        qlib_dir = os.path.join(os.getcwd(), "data", f"qlib_{self.region}")
        
        # Fetch and save data
        fetch_and_save_to_qlib(
            stock_symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            qlib_dir=qlib_dir,
            region=self.region,
            max_workers=16
        )


class QlibDumpData:
    @staticmethod
    def dump_yahoo_data(region="us", market="all", start_date="2015-01-01", end_date=None,
                       target_dir=None, symbols=None, max_workers=16, track_mlflow=True):
        """Dump Yahoo Finance data to Qlib format using Qlib's built-in tools"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        region = region.lower()
        qlib_region = REG_CN if region == "cn" else REG_US
        
        target_dir = target_dir or os.path.join(os.getcwd(), "data", f"qlib_{region}")
        
        print(f"Dumping Yahoo data for region {region}, market {market} to {target_dir}")
        
        # Log to MLflow if active
        if track_mlflow:
            try:
                # Use nested=True to avoid conflicts with parent run
                with mlflow.start_run(nested=True):
                    mlflow.log_param("qlib_dump_region", region)
                    mlflow.log_param("qlib_dump_market", market)
                    mlflow.log_param("qlib_dump_start_date", start_date)
                    mlflow.log_param("qlib_dump_end_date", end_date)
            except Exception as e:
                print(f"Error logging to MLflow: {str(e)}")
        
        try:
            # Fetch and save data using real implementation
            fetch_and_save_to_qlib(
                stock_symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                qlib_dir=target_dir,
                region=region,
                max_workers=max_workers,
                market=market
            )
            
            # Log success to MLflow
            if track_mlflow:
                mlflow.log_param("dump_status", "success")
                
            print(f"Data dumped successfully to {target_dir}")
            return target_dir
            
        except Exception as e:
            print(f"Error dumping data: {str(e)}")
            
            # Log error to MLflow
            if track_mlflow:
                mlflow.log_param("dump_status", "error")
                mlflow.log_param("dump_error", str(e))
            
            raise e
        finally:
            # Close MLflow run if active
            if track_mlflow:
                close_mlflow()
    
    @staticmethod
    def check_qlib_data(qlib_dir):
        """Check if Qlib data exists and is valid"""
        return exists_qlib_data(qlib_dir)


def dump_yahoo_data(stock_symbols=None, start_date=None, end_date=None, interval="1d", qlib_dir=None, region=None, max_workers=16, market=None):
    """Wrapper for QlibDumpData.dump_yahoo_data"""
    dumper = QlibDumpData()
    return dumper.dump_yahoo_data(
        region=region,
        market=market or "all",
        start_date=start_date,
        end_date=end_date,
        target_dir=qlib_dir,
        symbols=stock_symbols,
        max_workers=max_workers
    )


def fetch_data(region="cn", market="csi300", start_date="2015-01-01", end_date=None, qlib_dir=None, track=True):
    """
    Main fetch data function to be called from the workflow
    
    Parameters
    ----------
    region : str
        Region to fetch data for (cn or us)
    market : str
        Market to fetch data for (e.g., csi300, sp500)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str or None
        End date in YYYY-MM-DD format, or None for today
    qlib_dir : str or None
        Directory with Qlib data (will fetch if not provided)
    track : bool
        Whether to track with MLflow
        
    Returns
    -------
    str
        Path to the Qlib data directory
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Set up default qlib_dir if not provided
    if qlib_dir is None:
        qlib_dir = os.path.join(os.getcwd(), "data", f"qlib_{region}_{market}")
    
    print(f"Fetching data for {region} {market} from {start_date} to {end_date}")
    
    # Initialize MLflow if tracking is enabled
    if track:
        try:
            # Use nested=True to avoid conflicts with parent run
            with mlflow.start_run(nested=True):
                mlflow.log_param("fetch_region", region)
                mlflow.log_param("fetch_market", market)
                mlflow.log_param("fetch_start_date", start_date)
                mlflow.log_param("fetch_end_date", end_date)
                mlflow.log_param("fetch_qlib_dir", qlib_dir)
        except Exception as e:
            print(f"Error logging to MLflow: {str(e)}")
    
    # Get symbols for the specified market
    symbols = get_market_symbols(market, region)
    
    # Fetch and save data
    fetch_and_save_to_qlib(
        stock_symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        qlib_dir=qlib_dir,
        region=region,
        market=market
    )
    
    return qlib_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch historical stock data")
    parser.add_argument("--region", type=str, default="cn", choices=["cn", "us"], 
                      help="Region to fetch data for (cn or us)")
    parser.add_argument("--market", type=str, default="csi300", 
                      help="Market to fetch data for (e.g., csi300, sp500)")
    parser.add_argument("--start-date", type=str, default="2015-01-01", 
                      help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None, 
                      help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--qlib-dir", type=str, default=None, 
                      help="Directory to save Qlib data")
    parser.add_argument("--max-workers", type=int, default=16, 
                      help="Maximum number of worker processes")
    parser.add_argument("--track", action="store_true", default=True,
                      help="Whether to track with MLflow")
    parser.add_argument("--use-yfinance", action="store_true", default=False,
                      help="Use YahooDataFetcher instead of Qlib dumper")
    parser.add_argument("--symbols-file", type=str, default=None,
                      help="Path to file containing symbols (one per line)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load symbols from file if provided
    symbols = None
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    
    if args.use_yfinance:
        # Use YahooDataFetcher
        fetcher = YahooDataFetcher(region=args.region)
        fetcher.fetch_data(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols_file=args.symbols_file,
            track_mlflow=args.track
        )
    else:
        # Use Qlib dumper
        fetch_data(
            region=args.region,
            market=args.market if not symbols else None,  # Only use market if no symbols provided
            start_date=args.start_date,
            end_date=args.end_date,
            qlib_dir=args.qlib_dir,
            track=args.track
        )


class QlibDataFetcher:
    """
    Class for fetching and organizing data for Qlib
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data fetcher
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with the following keys:
            - region: str, Region code (cn/us)
            - start_date: str, Start date in YYYY-MM-DD format
            - end_date: str, End date in YYYY-MM-DD format
            - instruments: List[str], List of instrument symbols
            - exists_skip: bool, Whether to skip if data already exists
            - max_workers: int, Maximum number of workers for parallel processing
        """
        self.config = config or {}
        self.region = self.config.get("region", "cn")
        self.start_date = self.config.get("start_date", "2015-01-01")
        self.end_date = self.config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        self.instruments = self.config.get("instruments", None)
        self.exists_skip = self.config.get("exists_skip", True)
        self.max_workers = self.config.get("max_workers", 16)
        
    def fetch_data(self, qlib_dir: str, region: str = None, target_dir: str = "qlib_data", track_mlflow: bool = True, exists_skip: bool = None) -> str:
        """
        Fetch data and save it in Qlib format
        
        Parameters
        ----------
        qlib_dir : str
            Base directory for Qlib data
        region : str
            Region code (cn/us), overrides the one in config
        target_dir : str
            Target directory name under qlib_dir
        track_mlflow : bool
            Whether to track with MLflow
        exists_skip : bool
            Whether to skip if data already exists, overrides the one in config
            
        Returns
        -------
        str
            Path to the Qlib data directory
        """
        region = region or self.region
        exists_skip = exists_skip if exists_skip is not None else self.exists_skip
        
        # Set up paths
        qlib_dir = os.path.abspath(qlib_dir)
        data_path = os.path.join(qlib_dir, target_dir)
        os.makedirs(data_path, exist_ok=True)
        
        print(f"Fetching data for region {region} from {self.start_date} to {self.end_date}")
        
        # Check if data already exists
        if exists_skip and exists_qlib_data(data_path):
            print(f"Qlib data already exists at {data_path} and exists_skip=True, skipping download")
            return data_path
            
        # Initialize MLflow if tracking is enabled
        run_id = None
        if track_mlflow:
            run_id = init_mlflow(MLFLOW_CONFIG)
            mlflow.log_params({
                "qlib_data_path": data_path,
                "qlib_region": region,
                "qlib_start_date": self.start_date,
                "qlib_end_date": self.end_date
            })
            
        try:
            # Get market based on region
            market = "csi300" if region == "cn" else "sp500"
            
            # Use dump_yahoo_data function
            QlibDumpData.dump_yahoo_data(
                region=region,
                market=market,
                start_date=self.start_date,
                end_date=self.end_date,
                target_dir=data_path,
                symbols=self.instruments,
                max_workers=self.max_workers,
                track_mlflow=False  # We're handling MLflow tracking here
            )
            
            print(f"Data fetched and saved to {data_path}")
            
            # Log to MLflow if tracking is enabled
            if track_mlflow:
                mlflow.log_param("qlib_data_status", "success")
                
            return data_path
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            if track_mlflow:
                mlflow.log_param("qlib_data_status", "error")
                mlflow.log_param("qlib_data_error", str(e))
            raise e
        finally:
            if track_mlflow:
                close_mlflow()

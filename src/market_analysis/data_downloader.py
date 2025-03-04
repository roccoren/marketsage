"""
Stock Data Downloader Module

Downloads stock data from various sources and saves it in JSON format.
Currently supports Yahoo Finance as the data source.
"""
import os
import json
import datetime
import concurrent.futures
from typing import Dict, List, Optional, Union, Tuple
import yfinance as yf
import pandas as pd
from pathlib import Path


class StockDataDownloader:
    """
    Downloads stock data from various markets and saves it as JSON files.
    
    Currently supports NYSE, NASDAQ, ASX, and HKEX markets.
    Data is downloaded from Yahoo Finance by default.
    """
    
    # Market mappings for Yahoo Finance
    MARKET_CODES = {
        "NYSE": "",          # No suffix for NYSE stocks
        "NASDAQ": "",        # No suffix for NASDAQ stocks
        "ASX": ".AX",        # Australian Stock Exchange suffix
        "HKEX": ".HK"        # Hong Kong Stock Exchange suffix
    }
    
    def __init__(self, base_data_dir: str = None, max_workers: int = None):
        """
        Initialize the downloader with base directory for data storage
        
        Args:
            base_data_dir: Base directory for storing data. Defaults to "data/" 
                           in the project root directory
            max_workers: Maximum number of worker threads for parallel downloading.
                         If None, defaults to min(32, os.cpu_count() + 4)
        """
        if base_data_dir is None:
            # Default to the data directory in the project root
            self.base_data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.base_data_dir = Path(base_data_dir)
        
        # Create a cache directory for YFinance
        self.cache_dir = self.base_data_dir / "yfinance_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up yfinance cache
        yf.set_tz_cache_location(str(self.cache_dir))
        
        # Set up parallel processing parameters
        self.max_workers = max_workers
    
    def download_stock_data(self, 
                           ticker: str, 
                           market: str, 
                           start_date: Union[str, datetime.date],
                           end_date: Optional[Union[str, datetime.date]] = None,
                           save_dir: Optional[str] = None,
                           save_daily: bool = True) -> Dict:
        """
        Download stock data for a specific ticker and date range
        
        Args:
            ticker: Stock ticker symbol
            market: Market name (NYSE, NASDAQ, ASX, HKEX)
            start_date: Start date for data download (YYYY-MM-DD)
            end_date: Optional end date for data download (YYYY-MM-DD), defaults to today
            save_dir: Optional directory to save the data, defaults to market-specific directory
            save_daily: Whether to save individual daily files (default: True)
            
        Returns:
            Dict containing the stock data
            
        Raises:
            ValueError: If market is not supported
            RuntimeError: If data download fails
        """
        if market not in self.MARKET_CODES:
            raise ValueError(f"Unsupported market: {market}. " 
                           f"Supported markets are {list(self.MARKET_CODES.keys())}")
        
        # Format ticker symbol with market suffix if needed
        market_suffix = self.MARKET_CODES[market]
        full_ticker = f"{ticker}{market_suffix}"
        
        # Convert date strings to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            
        if end_date is None:
            end_date = datetime.date.today()
        elif isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        try:
            # Download data from Yahoo Finance
            stock = yf.Ticker(full_ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise RuntimeError(f"No data retrieved for {ticker} ({market})")
            
            # Set up market directory
            if save_dir is None:
                market_dir = self.base_data_dir / market
            else:
                market_dir = Path(save_dir)
            
            # Create market directory if it doesn't exist
            market_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ticker-specific subdirectory
            ticker_dir = market_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
                
            # Process and save each day's data separately
            results = {}
            if save_daily:
                for date, day_data in df.iterrows():
                    data_date = date.strftime("%Y-%m-%d")
                    
                    # Create data dictionary
                    history_item = {
                        "date": data_date,
                        "open": float(day_data["Open"]),
                        "high": float(day_data["High"]),
                        "low": float(day_data["Low"]),
                        "close": float(day_data["Close"]),
                        "volume": float(day_data["Volume"])
                    }
                    
                    # Save as JSON file in the ticker's directory only
                    daily_filename = f"{data_date}.json"
                    daily_path = ticker_dir / daily_filename
                    
                    with open(daily_path, 'w') as f:
                        json.dump({"history": [history_item]}, f, indent=2)
                    
                    # Add to results
                    results[data_date] = history_item
                
            # Also save the full historical data in the standard format
            full_data = {"history": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(day_data["Open"]),
                    "high": float(day_data["High"]),
                    "low": float(day_data["Low"]),
                    "close": float(day_data["Close"]),
                    "volume": float(day_data["Volume"])
                } for date, day_data in df.iterrows()
            ]}
            
            # Save complete history to ticker-specific folder
            ticker_file = ticker_dir / "history.json"
            with open(ticker_file, 'w') as f:
                json.dump(full_data, f, indent=2)
                
            return full_data
                
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {ticker} ({market}): {str(e)}")
    
    def _download_stock_worker(self, args: Tuple[str, str, Union[str, datetime.date], 
                                               Optional[Union[str, datetime.date]], Optional[str], bool]) -> Tuple[str, Dict, Exception]:
        """
        Worker function for parallel downloading
        
        Args:
            args: Tuple containing (ticker, market, start_date, end_date, save_dir, save_daily)
            
        Returns:
            Tuple of (ticker, data_dict, None) or (ticker, None, exception)
        """
        ticker, market, start_date, end_date, save_dir, save_daily = args
        try:
            data = self.download_stock_data(ticker, market, start_date, end_date, save_dir, save_daily)
            return ticker, data, None
        except Exception as e:
            return ticker, None, e
    
    def download_multiple_stocks(self,
                                tickers: List[str],
                                market: str,
                                start_date: Union[str, datetime.date],
                                end_date: Optional[Union[str, datetime.date]] = None,
                                save_dir: Optional[str] = None,
                                save_daily: bool = True,
                                parallel: bool = True,
                                batch_size: int = 10) -> Dict[str, Dict]:
        """
        Download data for multiple stocks in the same market
        
        Args:
            tickers: List of ticker symbols
            market: Market name (NYSE, NASDAQ, ASX, HKEX)
            start_date: Start date for data download
            end_date: Optional end date for data download, defaults to today
            save_dir: Optional directory to save the data
            save_daily: Whether to save individual daily files
            parallel: Whether to use parallel processing (default: True)
            batch_size: Number of stocks to download in each batch for rate limiting (default: 10)
            
        Returns:
            Dictionary mapping ticker symbols to their data
            
        Raises:
            ValueError: If market is not supported
        """
        results = {}
        
        # Validate market
        if market not in self.MARKET_CODES:
            raise ValueError(f"Unsupported market: {market}. " 
                           f"Supported markets are {list(self.MARKET_CODES.keys())}")
        
        if not parallel:
            # Sequential download
            for ticker in tickers:
                try:
                    print(f"Downloading data for {ticker} ({market})...")
                    data = self.download_stock_data(ticker, market, start_date, end_date, save_dir, save_daily)
                    results[ticker] = data
                    print(f"Successfully downloaded data for {ticker}")
                except Exception as e:
                    print(f"Error downloading {ticker}: {str(e)}")
        else:
            # Parallel download with batching
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch_tickers)} stocks)")
                
                worker_args = [(ticker, market, start_date, end_date, save_dir, save_daily) for ticker in batch_tickers]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_ticker = {executor.submit(self._download_stock_worker, args): args[0] for args in worker_args}
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker, data, error = future.result()
                        if error:
                            print(f"Error downloading {ticker}: {str(error)}")
                        else:
                            results[ticker] = data
                            print(f"Successfully downloaded data for {ticker}")
                
                # If not the last batch, add a small delay to prevent rate limiting
                if i + batch_size < len(tickers):
                    import time
                    time.sleep(1)  # Small delay between batches
        
        return results
    
    def download_market_list(self, 
                            market: str,
                            list_file: Optional[str] = None,
                            start_date: Optional[Union[str, datetime.date]] = None,
                            end_date: Optional[Union[str, datetime.date]] = None,
                            parallel: bool = True,
                            batch_size: int = 10,
                            save_daily: bool = True) -> Dict[str, Dict]:
        """
        Download data for all stocks in a market listing file
        
        Args:
            market: Market name (NYSE, NASDAQ, ASX, HKEX)
            list_file: Optional path to a JSON file with a list of ticker symbols.
                       If not provided, will look for {market.lower()}_filtered_listings.json
                       in the base data directory
            start_date: Start date for data download (default: 30 days ago)
            end_date: End date for data download (default: today)
            parallel: Whether to use parallel processing (default: True)
            batch_size: Number of stocks to download in each batch (default: 10)
            save_daily: Whether to save individual daily files (default: True)
                       
        Returns:
            Dictionary mapping ticker symbols to their data
            
        Raises:
            ValueError: If market is not supported or list file not found
        """
        if list_file is None:
            list_file = self.base_data_dir / f"{market.lower()}_filtered_listings.json"
        
        list_file = Path(list_file)
        
        if not list_file.exists():
            raise ValueError(f"Stock list file not found: {list_file}")
            
        try:
            with open(list_file, 'r') as f:
                listings = json.load(f)
            
            tickers = [listing["symbol"] for listing in listings]
            
            # Default to downloading the last 30 days if not specified
            if start_date is None:
                start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            
            return self.download_multiple_stocks(
                tickers, market, start_date, end_date, 
                parallel=parallel, batch_size=batch_size, save_daily=save_daily
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to download market data: {str(e)}")
    
    def batch_download_markets(self, 
                             markets: List[str], 
                             start_date: Optional[Union[str, datetime.date]] = None,
                             end_date: Optional[Union[str, datetime.date]] = None,
                             parallel: bool = True,
                             batch_size: int = 10,
                             save_daily: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Download data for multiple markets
        
        Args:
            markets: List of market names (NYSE, NASDAQ, ASX, HKEX)
            start_date: Start date for data download (default: 30 days ago)
            end_date: End date for data download (default: today)
            parallel: Whether to use parallel processing (default: True)
            batch_size: Number of stocks to download in each batch (default: 10)
            save_daily: Whether to save individual daily files (default: True)
                       
        Returns:
            Dictionary mapping markets to dictionaries of ticker data
            
        Raises:
            ValueError: If any market is not supported
        """
        results = {}
        
        for market in markets:
            if market not in self.MARKET_CODES:
                raise ValueError(f"Unsupported market: {market}. " 
                               f"Supported markets are {list(self.MARKET_CODES.keys())}")
        
        for market in markets:
            try:
                print(f"\n===== Downloading data for {market} market =====\n")
                market_data = self.download_market_list(
                    market, start_date=start_date, end_date=end_date,
                    parallel=parallel, batch_size=batch_size, save_daily=save_daily
                )
                results[market] = market_data
                print(f"\n===== Completed downloading {market} market data =====\n")
            except Exception as e:
                print(f"Error downloading {market} market data: {str(e)}")
        
        return results
        
    def download_from_json_files(self,
                               file_paths: List[str],
                               start_date: Optional[Union[str, datetime.date]] = None,
                               end_date: Optional[Union[str, datetime.date]] = None,
                               parallel: bool = True,
                               batch_size: int = 10,
                               save_daily: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Download data for stocks listed in JSON files containing ticker symbols
        
        Args:
            file_paths: List of JSON file paths containing stock listings
            start_date: Start date for data download (default: 30 days ago)
            end_date: End date for data download (default: today)
            parallel: Whether to use parallel processing (default: True)
            batch_size: Number of stocks to download in each batch (default: 10)
            save_daily: Whether to save individual daily files (default: True)
                       
        Returns:
            Dictionary mapping markets to dictionaries of ticker data
            
        Raises:
            ValueError: If file not found or has incorrect format
            
        Example:
            download_from_json_files(
                ['data/nyse-listed.json', 'data/nasdaq-listed.json'],
                start_date='2023-01-01',
                parallel=True,
                batch_size=20
            )
        """
        results = {}
        
        # Default to downloading the last 30 days if not specified
        if start_date is None:
            start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                
                if not file_path.exists():
                    print(f"Warning: File {file_path} not found, skipping")
                    continue
                
                # Determine market from filename
                if "nyse" in file_path.name.lower():
                    market = "NYSE"
                elif "nasdaq" in file_path.name.lower():
                    market = "NASDAQ"
                else:
                    market_name = file_path.stem.split("-")[0].split("_")[0].upper()
                    if market_name in self.MARKET_CODES:
                        market = market_name
                    else:
                        raise ValueError(f"Cannot determine market from filename: {file_path.name}")
                
                print(f"\n===== Loading symbols from {file_path} for {market} market =====\n")
                
                # Load listings from JSON file
                with open(file_path, 'r') as f:
                    listings = json.load(f)
                
                if not isinstance(listings, list):
                    raise ValueError(f"Expected list of symbols in {file_path}, got {type(listings)}")
                
                # Extract ticker symbols from the JSON structure
                # Support different formats: {"Symbol": "AAPL"} or {"symbol": "AAPL"}
                tickers = []
                for item in listings:
                    if "Symbol" in item:
                        tickers.append(item["Symbol"])
                    elif "symbol" in item:
                        tickers.append(item["symbol"])
                
                if not tickers:
                    print(f"No ticker symbols found in {file_path}")
                    continue
                
                print(f"Found {len(tickers)} ticker symbols in {file_path}")
                
                # Download data for the tickers
                market_data = self.download_multiple_stocks(
                    tickers, market, start_date, end_date,
                    parallel=parallel, batch_size=batch_size, save_daily=save_daily
                )
                
                results[market] = market_data
                print(f"\n===== Completed downloading data for {len(market_data)} {market} stocks =====\n")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return results

if __name__ == "__main__":
    # Example usage
    downloader = StockDataDownloader()
    
    # Download a single stock for the past 30 days
    start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    downloader.download_stock_data("AAPL", "NASDAQ", start_date)
    
    # Download multiple stocks in parallel
    # downloader.download_multiple_stocks(
    #     ["AAPL", "MSFT", "GOOGL"], "NASDAQ", start_date, parallel=True, batch_size=3
    # )
    
    # Download data for multiple markets
    # downloader.batch_download_markets(["NASDAQ", "NYSE"], start_date, parallel=True, batch_size=10)
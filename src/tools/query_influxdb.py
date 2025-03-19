#!/usr/bin/env python3
"""
Query InfluxDB for stock data.

This utility script allows you to query stock data from InfluxDB and export it to various formats.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from influxdb_client import InfluxDBClient
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default config path
CONFIG_DIR = Path("config")
STORAGE_CONFIG_FILE = CONFIG_DIR / "storage_config.json"

def load_config():
    """Load configuration from storage_config.json file."""
    try:
        if STORAGE_CONFIG_FILE.exists():
            with open(STORAGE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        else:
            logger.warning(f"Config file {STORAGE_CONFIG_FILE} not found, using default settings")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
    
    # Return default config
    return {
        "storage_type": "influxdb",
        "influxdb": {
            "url": "http://128.85.51.203:8086",
            "token": "RqYmYk85i3sxRFvTLBv46z8P1r5aVoPmn_sYyw9tUeXpmitpRDs6fSoirdm5eDrHOm_zEG8fJ6lRpbCQTOLGdw==",
            "org": "stockscan",
            "bucket": "stocks"
        },
        "file_storage": {
            "data_dir": "data",
            "ttl_days": 30
        }
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Query stock data from InfluxDB')
    
    # Query options
    parser.add_argument('--symbol', required=True, help='Stock symbol to query')
    parser.add_argument('--market', required=True, help='Market (e.g., nyse, nasdaq)')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD), defaults to today')
    
    # Output options
    parser.add_argument('--output-format', choices=['csv', 'json', 'plot'], default='plot',
                        help='Output format (csv, json, or plot)')
    parser.add_argument('--output-file', help='Output file path')
    parser.add_argument('--plot-type', choices=['line', 'candlestick', 'ohlc'], default='line',
                        help='Plot type (line, candlestick, or ohlc)')
    
    # Add verbose flag
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()

def query_data(client, bucket, symbol, market, start_date, end_date):
    """
    Query stock data from InfluxDB.
    
    Args:
        client: InfluxDBClient instance
        bucket: InfluxDB bucket name
        symbol: Stock symbol
        market: Market name
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with stock data
    """
    query_api = client.query_api()
    
    # Format dates for query
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Build Flux query
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: {start_date_str}T00:00:00Z, stop: {end_date_str}T23:59:59Z)
        |> filter(fn: (r) => r._measurement == "stock_data")
        |> filter(fn: (r) => r.symbol == "{symbol}")
        |> filter(fn: (r) => r.market == "{market}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    # Execute query
    tables = query_api.query(query)
    
    if not tables:
        logger.warning(f"No data found for {symbol} in {market} for the specified date range")
        return None
    
    # Convert to DataFrame
    records = []
    for table in tables:
        for record in table.records:
            record_dict = record.values
            records.append(record_dict)
    
    # If no records found, return None
    if not records:
        logger.warning(f"No data found for {symbol} in {market} for the specified date range")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Rename time column and set it as index
    if '_time' in df.columns:
        df['date'] = pd.to_datetime(df['_time'])
        df = df.set_index('date')
    
    # Filter only the columns we need
    columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'market_cap']
    df = df[[col for col in columns if col in df.columns]]
    
    # Sort by date
    df = df.sort_index()
    
    return df

def plot_data(df, symbol, market, plot_type):
    """
    Plot stock data.
    
    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
        market: Market name
        plot_type: Plot type (line, candlestick, or ohlc)
    """
    if plot_type == 'line':
        # Line plot of close prices
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price')
        
        if 'adjusted_close' in df.columns:
            plt.plot(df.index, df['adjusted_close'], label='Adjusted Close', linestyle='--')
            
        plt.title(f'{symbol} ({market}) - Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    elif plot_type in ['candlestick', 'ohlc']:
        # Import mplfinance for specialized financial plots
        try:
            import mplfinance as mpf
        except ImportError:
            logger.error("mplfinance package not installed. Install it with 'pip install mplfinance'")
            logger.info("Falling back to line plot")
            plot_data(df, symbol, market, 'line')
            return
        
        # Prepare data for mplfinance
        df_mpf = df[['open', 'high', 'low', 'close']].copy()
        
        # Add volume if available
        kwargs = {}
        if 'volume' in df.columns:
            kwargs['volume'] = df['volume']
            kwargs['type'] = 'candle' if plot_type == 'candlestick' else 'ohlc'
            kwargs['style'] = 'yahoo'
            kwargs['title'] = f'{symbol} ({market}) - OHLC Chart'
            kwargs['ylabel'] = 'Price'
            kwargs['ylabel_lower'] = 'Volume'
            kwargs['figratio'] = (12, 8)
            kwargs['figscale'] = 1.2
        
        # Plot
        mpf.plot(df_mpf, **kwargs)

def save_data(df, output_format, output_file, symbol, market):
    """
    Save data to file.
    
    Args:
        df: DataFrame with stock data
        output_format: Output format (csv or json)
        output_file: Output file path
        symbol: Stock symbol
        market: Market name
    """
    if not output_file:
        # Generate default filename
        start_date = df.index.min().strftime('%Y%m%d')
        end_date = df.index.max().strftime('%Y%m%d')
        output_file = f"{symbol}_{market}_{start_date}_{end_date}.{output_format}"
    
    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save file
    if output_format == 'csv':
        df.to_csv(output_file)
        logger.info(f"Data saved to {output_file}")
    elif output_format == 'json':
        # Convert dates to string for JSON serialization
        df.index = df.index.strftime('%Y-%m-%d')
        df.to_json(output_file, orient='index', indent=2)
        logger.info(f"Data saved to {output_file}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Load configuration
    config = load_config()
    
    if config["storage_type"] != "influxdb":
        logger.error("Storage type is not set to InfluxDB. Please run 'python main.py configure-storage --use-influxdb' first.")
        return
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Get InfluxDB settings
    influxdb_url = config["influxdb"]["url"]
    influxdb_token = config["influxdb"]["token"]
    influxdb_org = config["influxdb"]["org"]
    bucket_name = config["influxdb"]["bucket"]
    
    try:
        # Initialize InfluxDB client
        client = InfluxDBClient(
            url=influxdb_url,
            token=influxdb_token,
            org=influxdb_org
        )
        
        # Query data
        logger.info(f"Querying data for {args.symbol} in {args.market} from {start_date.date()} to {end_date.date()}")
        df = query_data(client, bucket_name, args.symbol, args.market, start_date, end_date)
        
        if df is None or df.empty:
            logger.error("No data found.")
            return
        
        # Process based on output format
        if args.output_format in ['csv', 'json']:
            save_data(df, args.output_format, args.output_file, args.symbol, args.market)
        elif args.output_format == 'plot':
            plot_data(df, args.symbol, args.market, args.plot_type)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main() 
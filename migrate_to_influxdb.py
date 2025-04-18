#!/usr/bin/env python3
"""
Migration script to transfer data from existing storage backends (CSV, SQLite, TSDB)
to InfluxDB.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from modules.storage import StorageManager
from modules.connection_config import get_influxdb_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate data from existing storage backends to InfluxDB"
    )
    parser.add_argument(
        "--source-backend",
        type=str,
        choices=["csv", "sqlite", "tsdb"],
        default="csv",
        help="Source storage backend type (default: csv)"
    )
    parser.add_argument(
        "--source-path",
        type=str,
        help="Path to source data directory or database file"
    )
    # Get default InfluxDB config
    influxdb_config = get_influxdb_config()
    
    parser.add_argument(
        "--influxdb-url",
        type=str,
        default=influxdb_config["url"],
        help=f"InfluxDB server URL (default: {influxdb_config['url']}). For remote server, use http://your-server-ip:8086"
    )
    parser.add_argument(
        "--influxdb-token",
        type=str,
        default=influxdb_config["token"],
        help="InfluxDB authentication token"
    )
    parser.add_argument(
        "--influxdb-org",
        type=str,
        default=influxdb_config["org"],
        help=f"InfluxDB organization name (default: {influxdb_config['org']})"
    )
    parser.add_argument(
        "--influxdb-bucket",
        type=str,
        default=influxdb_config["bucket"],
        help=f"InfluxDB bucket name (default: {influxdb_config['bucket']})"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to migrate (default: all symbols)"
    )
    parser.add_argument(
        "--skip-model-params",
        action="store_true",
        help="Skip migration of model parameters"
    )
    
    return parser.parse_args()

def migrate_stock_data(source_manager, target_manager, symbols=None, verbose=True):
    """
    Migrate stock data from source storage backend to target storage backend.
    
    Args:
        source_manager: Source StorageManager instance
        target_manager: Target StorageManager instance
        symbols: List of symbols to migrate (if None, migrate all symbols)
        verbose: Whether to show progress bar
    
    Returns:
        tuple: (success_count, error_count, total_count)
    """
    # Get list of symbols to migrate
    if symbols is None:
        symbols = source_manager.list_symbols()
    
    if not symbols:
        print("No symbols found in source storage")
        return 0, 0, 0
    
    print(f"Migrating stock data for {len(symbols)} symbols...")
    
    success_count = 0
    error_count = 0
    
    # Use tqdm for progress bar if verbose
    symbol_iter = tqdm(symbols) if verbose else symbols
    
    for symbol in symbol_iter:
        try:
            # Load data from source
            df = source_manager.load_data(symbol)
            
            if df.empty:
                print(f"No data found for symbol {symbol}")
                continue
            
            # Save data to target
            success = target_manager.save_data(symbol, df)
            
            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"Error saving data for symbol {symbol}")
        
        except Exception as e:
            error_count += 1
            print(f"Error migrating data for symbol {symbol}: {e}")
    
    return success_count, error_count, len(symbols)

def migrate_model_params(source_manager, target_manager, symbols=None, verbose=True):
    """
    Migrate model parameters from source storage backend to target storage backend.
    
    Args:
        source_manager: Source StorageManager instance
        target_manager: Target StorageManager instance
        symbols: List of symbols to migrate (if None, migrate all symbols)
        verbose: Whether to show progress bar
    
    Returns:
        tuple: (success_count, error_count, total_count)
    """
    # Get list of symbols to migrate
    if symbols is None:
        symbols = source_manager.list_symbols()
    
    if not symbols:
        print("No symbols found in source storage")
        return 0, 0, 0
    
    print(f"Migrating model parameters for {len(symbols)} symbols...")
    
    success_count = 0
    error_count = 0
    total_count = 0
    
    # Use tqdm for progress bar if verbose
    symbol_iter = tqdm(symbols) if verbose else symbols
    
    for symbol in symbol_iter:
        try:
            # Get list of models for this symbol
            models = source_manager.list_models(symbol)
            
            if not models:
                continue
            
            for model_name in models:
                try:
                    # Load model parameters from source
                    params = source_manager.load_model_params(symbol, model_name)
                    
                    if not params:
                        continue
                    
                    # Generate version ID based on timestamp
                    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}_{model_name}"
                    
                    # Save model parameters to target
                    success = target_manager.save_model_params(
                        symbol=symbol,
                        model_name=model_name,
                        params=params,
                        version=version,
                        description=f"Migrated from {source_manager.backend_type}"
                    )
                    
                    total_count += 1
                    
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"Error saving model parameters for {symbol}/{model_name}")
                
                except Exception as e:
                    error_count += 1
                    print(f"Error migrating model parameters for {symbol}/{model_name}: {e}")
        
        except Exception as e:
            print(f"Error listing models for symbol {symbol}: {e}")
    
    return success_count, error_count, total_count

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize source storage manager
    source_kwargs = {}
    if args.source_path:
        if args.source_backend == "csv":
            source_kwargs["base_dir"] = args.source_path
        elif args.source_backend == "sqlite":
            source_kwargs["db_path"] = args.source_path
    
    source_manager = StorageManager(backend_type=args.source_backend, **source_kwargs)
    
    # Initialize target storage manager (InfluxDB)
    # Use command-line arguments if provided, otherwise use config values
    target_manager = StorageManager(
        backend_type="influxdb",
        url=args.influxdb_url,
        token=args.influxdb_token,
        org=args.influxdb_org,
        bucket=args.influxdb_bucket
    )
    
    # Parse symbols list if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Migrate stock data
    data_success, data_error, data_total = migrate_stock_data(
        source_manager, target_manager, symbols, verbose=True
    )
    
    print(f"\nStock data migration complete:")
    print(f"  Success: {data_success}/{data_total} ({data_success/data_total*100:.1f}%)")
    print(f"  Errors: {data_error}/{data_total} ({data_error/data_total*100:.1f}%)")
    
    # Migrate model parameters if not skipped
    if not args.skip_model_params:
        model_success, model_error, model_total = migrate_model_params(
            source_manager, target_manager, symbols, verbose=True
        )
        
        if model_total > 0:
            print(f"\nModel parameters migration complete:")
            print(f"  Success: {model_success}/{model_total} ({model_success/model_total*100:.1f}%)")
            print(f"  Errors: {model_error}/{model_total} ({model_error/model_total*100:.1f}%)")
        else:
            print("\nNo model parameters found to migrate")
    
    print("\nMigration complete!")

if __name__ == "__main__":
    main()
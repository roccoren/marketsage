#!/usr/bin/env python3
"""
Configure storage backend for stock data.

This utility script allows you to configure whether to use InfluxDB or file-based storage
for stock market data in the stockscan system.
"""

import argparse
import json
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default config path
CONFIG_DIR = Path("config")
STORAGE_CONFIG_FILE = CONFIG_DIR / "storage_config.json"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Configure storage backend for stock data')
    
    # Main options
    parser.add_argument(
        '--use-influxdb', 
        action='store_true',
        help='Use InfluxDB for stock data storage'
    )
    parser.add_argument(
        '--use-file-storage', 
        action='store_true',
        help='Use file-based storage for stock data'
    )
    
    # InfluxDB configuration
    parser.add_argument(
        '--influxdb-url', 
        type=str,
        default="http://128.85.51.203:8086",
        help='InfluxDB server URL'
    )
    parser.add_argument(
        '--influxdb-token', 
        type=str,
        help='InfluxDB authentication token'
    )
    parser.add_argument(
        '--influxdb-org', 
        type=str,
        default="stockscan",
        help='InfluxDB organization'
    )
    parser.add_argument(
        '--influxdb-bucket', 
        type=str,
        default="stocks",
        help='InfluxDB bucket name'
    )
    
    return parser.parse_args()

def create_default_config():
    """Create default storage configuration."""
    return {
        "storage_type": "file",  # "file" or "influxdb"
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

def read_config():
    """Read storage configuration from file."""
    if not STORAGE_CONFIG_FILE.exists():
        # Create default config
        config = create_default_config()
        save_config(config)
        return config
    
    try:
        with open(STORAGE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error reading config file: {str(e)}")
        logger.info("Creating default configuration")
        config = create_default_config()
        save_config(config)
        return config

def save_config(config):
    """Save storage configuration to file."""
    # Create config directory if it doesn't exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(STORAGE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {STORAGE_CONFIG_FILE}")
        
        # Set permissions to read/write for user only
        os.chmod(STORAGE_CONFIG_FILE, 0o600)
    except Exception as e:
        logger.error(f"Error saving config file: {str(e)}")

def update_config(args):
    """Update configuration based on command line arguments."""
    config = read_config()
    
    # Determine storage type
    if args.use_influxdb:
        config["storage_type"] = "influxdb"
        logger.info("Setting storage type to InfluxDB")
    elif args.use_file_storage:
        config["storage_type"] = "file"
        logger.info("Setting storage type to file-based")
    
    # Update InfluxDB settings if provided
    if args.influxdb_url:
        config["influxdb"]["url"] = args.influxdb_url
    
    if args.influxdb_token:
        config["influxdb"]["token"] = args.influxdb_token
    
    if args.influxdb_org:
        config["influxdb"]["org"] = args.influxdb_org
    
    if args.influxdb_bucket:
        config["influxdb"]["bucket"] = args.influxdb_bucket
    
    # Save updated config
    save_config(config)
    return config

def display_current_config():
    """Display the current storage configuration."""
    config = read_config()
    
    logger.info("Current Storage Configuration:")
    logger.info(f"  Storage Type: {config['storage_type']}")
    
    if config["storage_type"] == "influxdb":
        logger.info("  InfluxDB Settings:")
        logger.info(f"    URL: {config['influxdb']['url']}")
        logger.info(f"    Organization: {config['influxdb']['org']}")
        logger.info(f"    Bucket: {config['influxdb']['bucket']}")
        # Don't log the token for security reasons
        logger.info(f"    Token: {'*' * 10}")
    else:
        logger.info("  File Storage Settings:")
        logger.info(f"    Data Directory: {config['file_storage']['data_dir']}")
        logger.info(f"    TTL Days: {config['file_storage']['ttl_days']}")

def validate_influxdb_connection(config):
    """Validate InfluxDB connection with current settings."""
    if config["storage_type"] != "influxdb":
        return
    
    try:
        from influxdb_client import InfluxDBClient
        
        logger.info("Testing InfluxDB connection...")
        client = InfluxDBClient(
            url=config["influxdb"]["url"],
            token=config["influxdb"]["token"],
            org=config["influxdb"]["org"]
        )
        
        # Check if we can connect
        health = client.health()
        logger.info(f"InfluxDB connection successful - status: {health.status}")
        
        # Check if the bucket exists
        buckets_api = client.buckets_api()
        existing_buckets = buckets_api.find_buckets().buckets
        bucket_exists = any(bucket.name == config["influxdb"]["bucket"] for bucket in existing_buckets)
        
        if bucket_exists:
            logger.info(f"Bucket '{config['influxdb']['bucket']}' exists")
        else:
            logger.warning(f"Bucket '{config['influxdb']['bucket']}' does not exist")
            logger.info("Creating bucket...")
            buckets_api.create_bucket(
                bucket_name=config["influxdb"]["bucket"],
                org=config["influxdb"]["org"]
            )
            logger.info(f"Bucket '{config['influxdb']['bucket']}' created successfully")
        
        client.close()
    except ImportError:
        logger.error("influxdb-client package not installed. Please install it using pip install influxdb-client")
    except Exception as e:
        logger.error(f"Error connecting to InfluxDB: {str(e)}")
        logger.warning("Falling back to file-based storage")
        
        # Update config to use file-based storage
        config["storage_type"] = "file"
        save_config(config)

def main():
    """Main entry point."""
    args = parse_args()
    
    # If no arguments provided, just display current config
    if not any([args.use_influxdb, args.use_file_storage, 
                args.influxdb_url != "http://128.85.51.203:8086",
                args.influxdb_token, args.influxdb_org != "stockscan",
                args.influxdb_bucket != "stocks"]):
        display_current_config()
        return
    
    # Update configuration
    config = update_config(args)
    
    # Display updated configuration
    display_current_config()
    
    # Test InfluxDB connection if using InfluxDB
    if config["storage_type"] == "influxdb":
        validate_influxdb_connection(config)

if __name__ == "__main__":
    main() 
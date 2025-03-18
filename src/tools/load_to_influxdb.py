#!/usr/bin/env python3

import os
import json
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import logging
import time
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('influxdb_load.log')
    ]
)
logger = logging.getLogger(__name__)

# InfluxDB configuration
INFLUXDB_URL = "http://128.85.51.203:8086"
INFLUXDB_TOKEN = "RqYmYk85i3sxRFvTLBv46z8P1r5aVoPmn_sYyw9tUeXpmitpRDs6fSoirdm5eDrHOm_zEG8fJ6lRpbCQTOLGdw=="
INFLUXDB_ORG = "stockscan"
BUCKET_NAME = "stocks"

# Processing configuration
BATCH_SIZE = 1000  # Number of points to write in one batch
MAX_RETRIES = 3    # Maximum number of retries for failed operations
RETRY_DELAY = 1    # Delay between retries in seconds
MAX_WORKERS = 4    # Number of parallel workers for processing files

def validate_stock_data(data: Dict[str, Any], file_path: str) -> bool:
    """Validate the stock data before processing."""
    required_fields = ['symbol', 'market', 'open', 'high', 'low', 'close', 
                      'volume', 'adjusted_close', 'date']
    
    # Check for required fields
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field '{field}' in {file_path}")
            return False
    
    # Validate numeric fields
    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
    for field in numeric_fields:
        try:
            float(data[field])
            # Basic sanity checks
            if float(data[field]) < 0:
                logger.error(f"Negative value for {field} in {file_path}")
                return False
        except (ValueError, TypeError):
            logger.error(f"Invalid numeric value for {field} in {file_path}")
            return False
    
    # Validate date format
    try:
        datetime.strptime(data['date'], '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format in {file_path}")
        return False
    
    return True

def create_bucket(client: InfluxDBClient) -> None:
    """Create the stocks bucket if it doesn't exist."""
    for attempt in range(MAX_RETRIES):
        try:
            buckets_api = client.buckets_api()
            existing_buckets = buckets_api.find_buckets().buckets
            bucket_exists = any(bucket.name == BUCKET_NAME for bucket in existing_buckets)
            
            if not bucket_exists:
                buckets_api.create_bucket(
                    bucket_name=BUCKET_NAME,
                    org=INFLUXDB_ORG
                )
                logger.info(f"Created bucket: {BUCKET_NAME}")
            else:
                logger.info(f"Bucket {BUCKET_NAME} already exists")
            return
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} creating bucket: {str(e)}")
            time.sleep(RETRY_DELAY)

def process_json_file(file_path: str, exchange: str) -> List[Point]:
    """Process a JSON file and convert it to InfluxDB point."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not validate_stock_data(data, file_path):
            return []
        
        # Create a point from the JSON data
        point = Point("stock_data") \
            .tag("symbol", data['symbol']) \
            .tag("exchange", exchange) \
            .tag("market", data['market']) \
            .field("open", float(data['open'])) \
            .field("high", float(data['high'])) \
            .field("low", float(data['low'])) \
            .field("close", float(data['close'])) \
            .field("volume", float(data['volume'])) \
            .field("adjusted_close", float(data['adjusted_close']))
        
        # Add market_cap if available and not null
        if 'market_cap' in data and data['market_cap'] is not None:
            try:
                market_cap = float(data['market_cap'])
                if market_cap >= 0:  # Only add if non-negative
                    point.field("market_cap", market_cap)
            except (ValueError, TypeError):
                logger.warning(f"Invalid market_cap value in {file_path}, skipping")
        
        point.time(datetime.strptime(data['date'], '%Y-%m-%d'))
        
        return [point]
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def write_batch(write_api, points: List[Point], exchange: str) -> bool:
    """Write a batch of points with retry mechanism."""
    for attempt in range(MAX_RETRIES):
        try:
            write_api.write(bucket=BUCKET_NAME, record=points)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to write batch for {exchange} after {MAX_RETRIES} attempts: {str(e)}")
                return False
            logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} writing batch for {exchange}: {str(e)}")
            time.sleep(RETRY_DELAY)
    return False

def process_exchange(exchange: str, client: InfluxDBClient) -> int:
    """Process all files for a single exchange."""
    exchange_dir = os.path.join('data', exchange)
    if not os.path.exists(exchange_dir):
        logger.warning(f"Directory not found: {exchange_dir}")
        return 0

    # Find all JSON files
    json_files = []
    for root, _, files in os.walk(exchange_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    logger.info(f"Found {len(json_files)} JSON files in {exchange}")
    
    # Process files in parallel
    points_batch = []
    total_points = 0
    failed_files = []
    processed_files = 0
    
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a progress bar
            with tqdm(total=len(json_files), desc=f"Processing {exchange}", unit="files") as pbar:
                # Process files in parallel
                future_to_file = {executor.submit(process_json_file, f, exchange): f for f in json_files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        points = future.result()
                        if points:
                            points_batch.extend(points)
                            processed_files += 1
                            
                            # Write batch if it reaches the batch size
                            if len(points_batch) >= BATCH_SIZE:
                                if write_batch(write_api, points_batch, exchange):
                                    total_points += len(points_batch)
                                    logger.debug(f"Successfully wrote batch of {len(points_batch)} points for {exchange}")
                                else:
                                    failed_files.append(file_path)
                                points_batch = []
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        failed_files.append(file_path)
                    
                    pbar.update(1)
        
        # Write remaining points
        if points_batch:
            if write_batch(write_api, points_batch, exchange):
                total_points += len(points_batch)
                logger.debug(f"Successfully wrote final batch of {len(points_batch)} points for {exchange}")
    
    finally:
        write_api.close()
    
    # Log summary for this exchange
    logger.info(f"Exchange {exchange} summary:")
    logger.info(f"  Total files found: {len(json_files)}")
    logger.info(f"  Successfully processed files: {processed_files}")
    logger.info(f"  Total points written: {total_points}")
    logger.info(f"  Failed files: {len(failed_files)}")
    
    # Log failed files
    if failed_files:
        logger.error(f"Failed to process {len(failed_files)} files for {exchange}")
        with open(f"failed_files_{exchange}.txt", "w") as f:
            f.write("\n".join(failed_files))
    
    return total_points

def main():
    start_time = time.time()
    total_points_all_exchanges = 0
    
    # Initialize InfluxDB client
    client = InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG
    )
    
    try:
        # Create bucket if it doesn't exist
        create_bucket(client)
        
        # Process each exchange directory
        exchanges = ['asx', 'hkex', 'nyse', 'nasdaq']
        for exchange in exchanges:
            logger.info(f"Starting to process {exchange} data...")
            total_points = process_exchange(exchange, client)
            total_points_all_exchanges += total_points
            logger.info(f"Completed {exchange}. Total points written: {total_points}")
    
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
    finally:
        client.close()
        
        # Log summary
        duration = time.time() - start_time
        logger.info("\nFinal Summary:")
        logger.info(f"Total points written: {total_points_all_exchanges}")
        logger.info(f"Total time: {duration:.2f} seconds")
        logger.info(f"Average write rate: {total_points_all_exchanges/duration:.2f} points/second")

if __name__ == "__main__":
    main() 
"""InfluxDB based market data cache implementation."""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from functools import lru_cache
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS, SYNCHRONOUS
import pandas as pd

logger = logging.getLogger(__name__)

# Default InfluxDB configuration
DEFAULT_INFLUXDB_URL = "http://128.85.51.203:8086"
DEFAULT_INFLUXDB_TOKEN = "RqYmYk85i3sxRFvTLBv46z8P1r5aVoPmn_sYyw9tUeXpmitpRDs6fSoirdm5eDrHOm_zEG8fJ6lRpbCQTOLGdw=="
DEFAULT_INFLUXDB_ORG = "stockscan"
DEFAULT_BUCKET_NAME = "stocks"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
CONNECTION_TIMEOUT = 30  # seconds
MAX_CONNECTION_POOL_SIZE = 30  # Increased from 20 to 30 for even more concurrent connections
WRITE_BUFFER_SIZE = 80000  # Doubled from 40000 to 80000 for much larger batches
BATCH_SIZE = 200  # Doubled from 100 to 200 for more aggressive batching

# Config path
CONFIG_FILE = Path("config") / "storage_config.json"


def load_config():
    """Load configuration from storage_config.json file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        else:
            logger.warning(f"Config file {CONFIG_FILE} not found, using default settings")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
    
    # Return default config
    return {
        "storage_type": "influxdb",
        "influxdb": {
            "url": DEFAULT_INFLUXDB_URL,
            "token": DEFAULT_INFLUXDB_TOKEN,
            "org": DEFAULT_INFLUXDB_ORG,
            "bucket": DEFAULT_BUCKET_NAME,
            "max_connections": MAX_CONNECTION_POOL_SIZE,
            "write_buffer_size": WRITE_BUFFER_SIZE,
            "batch_size": BATCH_SIZE
        },
        "file_storage": {
            "data_dir": "data",
            "ttl_days": 30
        }
    }


class InfluxDBMarketDataCache:
    """Market data caching system using InfluxDB."""
    
    def __init__(self, data_dir: str = "data", ttl_days: int = 30):
        """
        Initialize InfluxDB cache.
        
        Args:
            data_dir: Fallback directory for file-based cache if InfluxDB is unavailable
            ttl_days: TTL for data retention
        """
        # Load configuration
        self.config = load_config()
        
        # Get InfluxDB settings from config
        self.influxdb_url = self.config["influxdb"]["url"]
        self.influxdb_token = self.config["influxdb"]["token"]
        self.influxdb_org = self.config["influxdb"]["org"]
        self.bucket_name = self.config["influxdb"]["bucket"]
        
        # Get advanced settings with defaults
        self.max_connections = self.config["influxdb"].get("max_connections", MAX_CONNECTION_POOL_SIZE)
        self.write_buffer_size = self.config["influxdb"].get("write_buffer_size", WRITE_BUFFER_SIZE)
        self.batch_size = self.config["influxdb"].get("batch_size", BATCH_SIZE)
        
        # Initialize InfluxDB client
        try:
            # Configure client with simpler parameters
            self.client = InfluxDBClient(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org,
                timeout=CONNECTION_TIMEOUT * 1000,  # milliseconds
                enable_gzip=True,  # Enable gzip compression
                debug=False,
                pool_size=self.max_connections
            )
            
            # Configure write options for better performance
            write_options = ASYNCHRONOUS
            write_options.batch_size = self.batch_size
            write_options.flush_interval = 1_000  # 1 second - more frequent flushes
            
            # Use ASYNCHRONOUS writing with simplified options
            self.write_api = self.client.write_api(write_options=write_options)
            self.query_api = self.client.query_api()
            
            # Create bucket if it doesn't exist
            self._create_bucket()
            logger.info(f"Successfully connected to InfluxDB at {self.influxdb_url} with pool size {self.max_connections}")
        except Exception as e:
            logger.error(f"Error connecting to InfluxDB: {str(e)}")
            logger.warning("Will use fallback file-based cache for all operations")
            self.client = None
            self.write_api = None
            self.query_api = None
        
        # Fallback to file-based cache if InfluxDB is unavailable
        self.fallback_cache_dir = Path(data_dir)
        self.fallback_cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
    
    def _create_bucket(self):
        """Create the InfluxDB bucket if it doesn't exist."""
        if not self.client:
            return
            
        for attempt in range(MAX_RETRIES):
            try:
                buckets_api = self.client.buckets_api()
                existing_buckets = buckets_api.find_buckets().buckets
                bucket_exists = any(bucket.name == self.bucket_name for bucket in existing_buckets)
                
                if not bucket_exists:
                    buckets_api.create_bucket(
                        bucket_name=self.bucket_name,
                        org=self.influxdb_org
                    )
                    logger.info(f"Created bucket: {self.bucket_name}")
                return
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to create InfluxDB bucket after {MAX_RETRIES} attempts: {str(e)}")
                else:
                    logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} creating bucket: {str(e)}")
                    asyncio.sleep(RETRY_DELAY)
    
    def _get_fallback_cache_path(self, market: str, symbol: str, date: datetime) -> Path:
        """Get fallback cache file path for market data."""
        return self.fallback_cache_dir / market / symbol / f"{date.strftime('%Y-%m-%d')}.json"
    
    async def get(self, market: str, symbol: str, date: datetime) -> Optional[dict]:
        """
        Get market data from InfluxDB.
        
        Args:
            market: Market name (e.g., "nasdaq", "nyse")
            symbol: Stock symbol
            date: Date for data retrieval
            
        Returns:
            Market data as dictionary or None if not found
        """
        # If client isn't available or config specifies file storage, use fallback cache
        if not self.client or self.config.get("storage_type") != "influxdb":
            return await self._get_from_fallback(market, symbol, date)
            
        try:
            # Format date strings for query
            date_str = date.strftime('%Y-%m-%d')
            next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Build Flux query
            query = f'''
            from(bucket: "{self.bucket_name}")
                |> range(start: {date_str}T00:00:00Z, stop: {next_day}T00:00:00Z)
                |> filter(fn: (r) => r._measurement == "stock_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.market == "{market}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            # Execute query
            tables = self.query_api.query(query)
            
            if not tables or len(tables) == 0:
                # Try fallback cache
                return await self._get_from_fallback(market, symbol, date)
            
            # Process results
            records = []
            for table in tables:
                for record in table.records:
                    record_dict = record.values
                    records.append(record_dict)
            
            if records:
                # Convert to the expected format
                data = {
                    "open": float(records[0].get('open', 0)),
                    "high": float(records[0].get('high', 0)),
                    "low": float(records[0].get('low', 0)),
                    "close": float(records[0].get('close', 0)),
                    "volume": float(records[0].get('volume', 0)),
                    "adjusted_close": float(records[0].get('adjusted_close', records[0].get('close', 0))),
                    "date": date_str,
                    "market": market,
                    "symbol": symbol
                }
                
                # Add optional fields if available
                for field in ['market_cap', 'sector', 'industry', 'fullTimeEmployees', 'longName', 'shortName']:
                    if field in records[0]:
                        data[field] = records[0][field]
                
                return data
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving data from InfluxDB for {symbol}: {str(e)}")
            # Try fallback cache
            return await self._get_from_fallback(market, symbol, date)
    
    async def _get_from_fallback(self, market: str, symbol: str, date: datetime) -> Optional[dict]:
        """Get data from fallback file cache."""
        cache_path = self._get_fallback_cache_path(market, symbol, date)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.loads(f.read())
        except Exception as e:
            logger.error(f"Error reading fallback cache for {symbol}: {str(e)}")
            return None
    
    async def set(self, market: str, symbol: str, date: datetime, data: dict):
        """
        Store market data in InfluxDB.
        
        Args:
            market: Market name
            symbol: Stock symbol
            date: Data date
            data: Market data dictionary
        """
        # Always write to fallback cache for reliability
        await self._set_to_fallback(market, symbol, date, data)
        
        # If client isn't available or config specifies file storage, stop here
        if not self.client or self.config.get("storage_type") != "influxdb":
            return
            
        try:
            # Create point
            point = Point("stock_data") \
                .tag("symbol", symbol) \
                .tag("market", market) \
                .field("open", float(data.get('open', 0))) \
                .field("high", float(data.get('high', 0))) \
                .field("low", float(data.get('low', 0))) \
                .field("close", float(data.get('close', 0))) \
                .field("volume", float(data.get('volume', 0))) \
                .field("adjusted_close", float(data.get('adjusted_close', data.get('close', 0))))
            
            # Add market_cap if available
            if 'market_cap' in data and data['market_cap'] is not None:
                try:
                    market_cap = float(data['market_cap'])
                    if market_cap >= 0:
                        point.field("market_cap", market_cap)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid market_cap value for {symbol}, skipping field")
            
            # Add additional fields if available
            for field in ['sector', 'industry', 'fullTimeEmployees', 'longName', 'shortName']:
                if field in data and data[field] is not None:
                    point.field(field, str(data[field]))
            
            # Set timestamp
            if isinstance(date, datetime):
                point.time(date)
            else:
                # Handle string dates
                try:
                    if 'date' in data and data['date']:
                        date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
                        point.time(date_obj)
                    else:
                        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        point.time(date_obj)
                except Exception as e:
                    # Default to current timestamp if date parsing fails
                    logger.warning(f"Failed to parse date for {symbol}: {str(e)}")
                    point.time(datetime.now())
            
            # Write to InfluxDB - with ASYNCHRONOUS write_api, no need to wait
            self.write_api.write(bucket=self.bucket_name, record=point)
            
        except Exception as e:
            logger.error(f"Error writing data to InfluxDB for {symbol}: {str(e)}")
    
    async def _set_to_fallback(self, market: str, symbol: str, date: datetime, data: dict):
        """Store data in fallback file cache."""
        cache_path = self._get_fallback_cache_path(market, symbol, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w') as f:
                f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error writing to fallback cache for {symbol}: {str(e)}")
    
    async def set_batch(self, batch_data: List):
        """
        Store multiple data points in InfluxDB as a batch operation.
        
        Args:
            batch_data: List of tuples (market, symbol, date, data)
        """
        # If client isn't available or config specifies file storage, use fallback
        if not self.client or self.config.get("storage_type") != "influxdb":
            for market, symbol, date, data in batch_data:
                await self._set_to_fallback(market, symbol, date, data)
            return
        
        try:
            # Create points for batch write
            points = []
            
            for market, symbol, date, data in batch_data:
                # Always write to fallback cache for reliability
                await self._set_to_fallback(market, symbol, date, data)
                
                # Create point
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .tag("market", market) \
                    .field("open", float(data.get('open', 0))) \
                    .field("high", float(data.get('high', 0))) \
                    .field("low", float(data.get('low', 0))) \
                    .field("close", float(data.get('close', 0))) \
                    .field("volume", float(data.get('volume', 0))) \
                    .field("adjusted_close", float(data.get('adjusted_close', data.get('close', 0))))
                
                # Add market_cap if available
                if 'market_cap' in data and data['market_cap'] is not None:
                    try:
                        market_cap = float(data['market_cap'])
                        if market_cap >= 0:
                            point.field("market_cap", market_cap)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid market_cap value for {symbol}, skipping field")
                
                # Add additional fields if available
                for field in ['sector', 'industry', 'fullTimeEmployees', 'longName', 'shortName']:
                    if field in data and data[field] is not None:
                        point.field(field, str(data[field]))
                
                # Set timestamp
                if isinstance(date, datetime):
                    point.time(date)
                else:
                    # Handle string dates
                    try:
                        if 'date' in data and data['date']:
                            date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
                            point.time(date_obj)
                        else:
                            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            point.time(date_obj)
                    except Exception as e:
                        # Default to current timestamp if date parsing fails
                        logger.warning(f"Failed to parse date for {symbol}: {str(e)}")
                        point.time(datetime.now())
                
                points.append(point)
            
            # Write batch to InfluxDB
            if points:
                for attempt in range(MAX_RETRIES):
                    try:
                        # Use non-blocking batch write
                        self.write_api.write(bucket=self.bucket_name, record=points)
                        logger.debug(f"Successfully wrote batch of {len(points)} points to InfluxDB")
                        break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            logger.error(f"Failed to write batch data to InfluxDB after {MAX_RETRIES} attempts: {str(e)}")
                        else:
                            logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} writing batch to InfluxDB: {str(e)}")
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
        
        except Exception as e:
            logger.error(f"Error writing batch data to InfluxDB: {str(e)}")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'client') and self.client:
            self.client.close() 
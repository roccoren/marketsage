#!/usr/bin/env python3
"""
InfluxDB storage backend for the stock analysis project.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import requests
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from modules.storage import StorageBackend
from modules.connection_config import get_influxdb_config
from modules.storage_backends.mock_influxdb import MockInfluxDBClient


class InfluxDBStorage(StorageBackend):
    """Storage backend using InfluxDB."""
    
    def __init__(self, url: str = None,
                 token: str = None,
                 org: str = None,
                 bucket: str = None):
        """
        Initialize InfluxDB storage.
        
        Args:
            url (str, optional): InfluxDB server URL. For remote server, use 'http://your-server-ip:8086'
            token (str, optional): Authentication token
            org (str, optional): Organization name
            bucket (str, optional): Bucket name (database)
            
        If any parameter is None, it will be loaded from the configuration.
        """
        # Get connection details directly from config file
        config = get_influxdb_config()
        
        # Use provided values or defaults from config
        self.url = url or config["url"]
        self.token = token or config["token"]
        self.org = org or config["org"]
        self.bucket = bucket or config["bucket"]
        
        # Print configuration for debugging
        print(f"InfluxDB Configuration: URL={self.url}, Org={self.org}, Bucket={self.bucket}")
        
        # Try to initialize real client, fall back to mock if unavailable
        try:
            
            print(f"Attempting to connect to InfluxDB at {self.url}")
            
            # Try to connect to the InfluxDB server
            try:
                response = requests.get(self.url, timeout=5)
                if response.status_code == 200:
                    # Connection successful, use real client
                    self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
                    self.is_mock = False
                    print(f"Connected to InfluxDB at {self.url}")
                else:
                    # Connection failed, use mock client
                    print(f"InfluxDB server responded with status code {response.status_code}, using mock client")
                    self.client = MockInfluxDBClient(url=self.url, token=self.token, org=self.org)
                    self.is_mock = True
            except Exception as conn_error:
                print(f"Could not connect to InfluxDB at {self.url}: {conn_error}")
                print("Using mock InfluxDB client instead")
                self.client = MockInfluxDBClient(url=self.url, token=self.token, org=self.org)
                self.is_mock = True
        except Exception as e:
            # Connection error, use mock client
            print(f"Error initializing InfluxDB connection: {e}")
            print("Using mock InfluxDB client instead")
            self.client = MockInfluxDBClient(url="http://localhost:8086", token="mock", org="mock")
            self.is_mock = True
            
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        # Ensure bucket exists
        self._init_db()
    
    def _init_db(self):
        """Initialize database if it doesn't exist."""
        try:
            buckets_api = self.client.buckets_api()
            bucket_list = buckets_api.find_buckets().buckets
            bucket_names = [bucket.name for bucket in bucket_list]
            
            if self.bucket not in bucket_names:
                buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)
                print(f"Created bucket: {self.bucket}")
            else:
                print(f"Using existing bucket: {self.bucket}")
        except Exception as e:
            print(f"Error initializing InfluxDB: {e}")
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol to InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        try:
            # Convert DataFrame to InfluxDB points
            points = []
            for timestamp, row in data.iterrows():
                point = Point("stock_data") \
                    .tag("symbol", symbol) \
                    .time(timestamp)
                
                # Add all numeric columns as fields
                for col, value in row.items():
                    if pd.api.types.is_numeric_dtype(type(value)):
                        point = point.field(col.lower(), float(value))
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(bucket=self.bucket, record=points)
            return True
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol from InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Build Flux query
            time_filter = ""
            if start_date:
                time_filter += f' and r._time >= time(v: "{start_date}T00:00:00Z")'
            if end_date:
                time_filter += f' and r._time <= time(v: "{end_date}T23:59:59Z")'
            
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -100y)
                |> filter(fn: (r) => r._measurement == "stock_data" and r.symbol == "{symbol}"{time_filter})
                |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            # Execute query
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                return pd.DataFrame()
            
            # Clean up result
            df = result.drop(columns=['result', 'table', '_start', '_stop', '_measurement', 'symbol'])
            df = df.rename(columns={
                '_time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Set date as index
            df = df.set_index('Date')
            df.index = pd.to_datetime(df.index)
            
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_model_params(self, symbol: str, model_name: str, params: Dict[str, Any], 
                         version: str = None, description: str = None,
                         metrics: Dict[str, Any] = None) -> bool:
        """
        Save model parameters to InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            params (Dict[str, Any]): Model parameters
            version (str, optional): Version identifier
            description (str, optional): Model description
            metrics (Dict[str, Any], optional): Model performance metrics
            
        Returns:
            bool: Success status
        """
        try:
            # Generate version ID if not provided
            if not version:
                version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Convert params to JSON string
            params_json = json.dumps(params, default=str)
            
            # Create point
            point = Point("model_params") \
                .tag("symbol", symbol) \
                .tag("model_name", model_name) \
                .tag("version", version) \
                .tag("status", "active") \
                .field("params", params_json)
            
            if description:
                point = point.field("description", description)
                
            if metrics:
                metrics_json = json.dumps(metrics, default=str)
                point = point.field("metrics", metrics_json)
                
                # Add individual metrics as separate fields for easier querying
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        point = point.field(f"metric_{key}", value)
            
            # Set timestamp
            point = point.time(datetime.utcnow())
            
            # Mark previous active versions as superseded
            self._update_version_status(symbol, model_name, "active", "superseded")
            
            # Write point to InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error saving model parameters for {symbol}/{model_name}: {e}")
            return False
    
    def load_model_params(self, symbol: str, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Load model parameters from InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            version (str, optional): Version to load (if None, loads active version)
            
        Returns:
            Dict[str, Any]: Model parameters
        """
        try:
            # Build query
            if version:
                query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: -100y)
                    |> filter(fn: (r) => r._measurement == "model_params" and 
                                       r.symbol == "{symbol}" and 
                                       r.model_name == "{model_name}" and
                                       r.version == "{version}")
                    |> last()
                '''
            else:
                query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: -100y)
                    |> filter(fn: (r) => r._measurement == "model_params" and 
                                       r.symbol == "{symbol}" and 
                                       r.model_name == "{model_name}" and
                                       r.status == "active")
                    |> last()
                '''
            
            # Execute query
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                return {}
            
            # Extract params JSON
            params_json = result['params'].iloc[0]
            return json.loads(params_json)
        except Exception as e:
            print(f"Error loading model parameters for {symbol}/{model_name}: {e}")
            return {}
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols in InfluxDB.
        
        Returns:
            List[str]: List of symbols
        """
        try:
            query = f'''
            import "influxdata/influxdb/schema"
            schema.tagValues(bucket: "{self.bucket}", tag: "symbol", predicate: (r) => r._measurement == "stock_data")
            '''
            
            result = self.query_api.query(query)
            
            if not result:
                return []
            
            symbols = [record.values.get('_value') for table in result for record in table.records]
            return symbols
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []
    
    def list_models(self, symbol: str) -> List[str]:
        """
        List all available models for a symbol in InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[str]: List of model names
        """
        try:
            query = f'''
            import "influxdata/influxdb/schema"
            schema.tagValues(
                bucket: "{self.bucket}",
                tag: "model_name",
                predicate: (r) => r._measurement == "model_params" and r.symbol == "{symbol}"
            )
            '''
            
            result = self.query_api.query(query)
            
            if not result:
                return []
            
            models = [record.values.get('_value') for table in result for record in table.records]
            return models
        except Exception as e:
            print(f"Error listing models for {symbol}: {e}")
            return []
            
    def list_model_versions(self, symbol: str, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model with their details.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Model name
            
        Returns:
            List[Dict[str, Any]]: List of version details
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -100y)
                |> filter(fn: (r) => r._measurement == "model_params" and 
                                   r.symbol == "{symbol}" and 
                                   r.model_name == "{model_name}")
                |> group(columns: ["version"])
                |> sort(columns: ["_time"], desc: true)
            '''
            
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                return []
            
            versions = []
            for _, row in result.iterrows():
                version_info = {
                    "version": row.get("version"),
                    "timestamp": row.get("_time"),
                    "status": row.get("status"),
                    "description": row.get("description")
                }
                
                # Extract metrics if available
                if "metrics" in row:
                    try:
                        metrics = json.loads(row["metrics"])
                        version_info.update(metrics)
                    except:
                        pass
                
                versions.append(version_info)
            
            return versions
        except Exception as e:
            print(f"Error listing model versions: {e}")
            return []
            
    def rollback_to_version(self, symbol: str, model_name: str, version: str) -> bool:
        """
        Rollback to a specific model version.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Model name
            version (str): Version to rollback to
            
        Returns:
            bool: Success status
        """
        try:
            # Get the specified version
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -100y)
                |> filter(fn: (r) => r._measurement == "model_params" and 
                                   r.symbol == "{symbol}" and 
                                   r.model_name == "{model_name}" and
                                   r.version == "{version}")
                |> last()
            '''
            
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                print(f"Version {version} not found")
                return False
            
            # Get the parameters from this version
            params_json = result['params'].iloc[0]
            
            # Get metrics if available
            metrics_json = None
            if 'metrics' in result.columns:
                metrics_json = result['metrics'].iloc[0]
            
            # Get description if available
            description = None
            if 'description' in result.columns:
                description = result['description'].iloc[0]
            
            # Mark current active version as superseded
            self._update_version_status(symbol, model_name, "active", "superseded")
            
            # Create a new point with the old parameters but marked as active
            point = Point("model_params") \
                .tag("symbol", symbol) \
                .tag("model_name", model_name) \
                .tag("version", f"{version}_restored") \
                .tag("status", "active") \
                .tag("restored_from", version) \
                .field("params", params_json) \
                .field("description", f"Restored from version {version}")
            
            if metrics_json:
                point = point.field("metrics", metrics_json)
            
            # Set timestamp
            point = point.time(datetime.utcnow())
            
            # Write point to InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error rolling back model: {e}")
            return False
            
    def _update_version_status(self, symbol: str, model_name: str, old_status: str, new_status: str) -> bool:
        """
        Update the status of model versions.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Model name
            old_status (str): Old status to match
            new_status (str): New status to set
            
        Returns:
            bool: Success status
        """
        try:
            # Find versions with the old status
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -100y)
                |> filter(fn: (r) => r._measurement == "model_params" and 
                                   r.symbol == "{symbol}" and 
                                   r.model_name == "{model_name}" and
                                   r.status == "{old_status}")
                |> last()
            '''
            
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                return True  # No versions to update
            
            # Update each version
            for _, row in result.iterrows():
                version = row.get("version")
                params_json = row.get("params")
                
                # Get metrics if available
                metrics_json = None
                if 'metrics' in result.columns:
                    metrics_json = row.get("metrics")
                
                # Get description if available
                description = None
                if 'description' in result.columns:
                    description = row.get("description", "")
                
                # Create a new point with updated status
                point = Point("model_params") \
                    .tag("symbol", symbol) \
                    .tag("model_name", model_name) \
                    .tag("version", version) \
                    .tag("status", new_status) \
                    .field("params", params_json)
                
                if description:
                    point = point.field("description", description)
                
                if metrics_json:
                    point = point.field("metrics", metrics_json)
                
                # Set timestamp
                point = point.time(datetime.utcnow())
                
                # Write point to InfluxDB
                self.write_api.write(bucket=self.bucket, record=point)
            
            return True
        except Exception as e:
            print(f"Error updating version status: {e}")
            return False
    
    def save_training_history(self, symbol: str, model_name: str, 
                            version: str, history: Dict[str, Any]) -> bool:
        """
        Save model training history to InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Model name
            version (str): Model version
            history (Dict[str, Any]): Training history (e.g., loss values by epoch)
            
        Returns:
            bool: Success status
        """
        try:
            # Convert history to JSON string
            history_json = json.dumps(history, default=str)
            
            # Create point
            point = Point("model_training_history") \
                .tag("symbol", symbol) \
                .tag("model_name", model_name) \
                .tag("version", version) \
                .field("history", history_json)
            
            # Add summary metrics if available
            if "loss" in history and len(history["loss"]) > 0:
                point = point.field("final_loss", history["loss"][-1])
                point = point.field("epochs", len(history["loss"]))
            
            if "val_loss" in history and len(history["val_loss"]) > 0:
                point = point.field("final_val_loss", history["val_loss"][-1])
            
            # Set timestamp
            point = point.time(datetime.utcnow())
            
            # Write point to InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            print(f"Error saving training history: {e}")
            return False
    
    def get_training_history(self, symbol: str, model_name: str, version: str) -> Dict[str, Any]:
        """
        Get model training history from InfluxDB.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Model name
            version (str): Model version
            
        Returns:
            Dict[str, Any]: Training history
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -100y)
                |> filter(fn: (r) => r._measurement == "model_training_history" and 
                                   r.symbol == "{symbol}" and 
                                   r.model_name == "{model_name}" and
                                   r.version == "{version}")
                |> last()
            '''
            
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                return {}
            
            # Extract history JSON
            history_json = result['history'].iloc[0]
            return json.loads(history_json)
        except Exception as e:
            print(f"Error getting training history: {e}")
            return {}
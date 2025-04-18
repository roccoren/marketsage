#!/usr/bin/env python3
"""
Mock implementation of InfluxDB client for testing purposes.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


class MockBucketsApi:
    """Mock implementation of InfluxDB BucketsApi."""
    
    def __init__(self):
        """Initialize with empty buckets."""
        self.buckets = []
    
    def find_buckets(self):
        """Return bucket list."""
        return MockBucketList(self.buckets)
    
    def create_bucket(self, bucket_name, org):
        """Create a new bucket."""
        self.buckets.append(MockBucket(bucket_name))
        print(f"Created mock bucket: {bucket_name}")


class MockBucket:
    """Mock implementation of InfluxDB Bucket."""
    
    def __init__(self, name):
        """Initialize with name."""
        self.name = name


class MockBucketList:
    """Mock implementation of InfluxDB BucketList."""
    
    def __init__(self, buckets):
        """Initialize with buckets."""
        self.buckets = buckets


class MockWriteApi:
    """Mock implementation of InfluxDB WriteApi."""
    
    def __init__(self):
        """Initialize with empty data store."""
        self.data = {}
    
    def write(self, bucket, record):
        """Write a record to the mock data store."""
        if bucket not in self.data:
            self.data[bucket] = []
        
        if isinstance(record, list):
            for point in record:
                self.data[bucket].append(self._point_to_dict(point))
        else:
            self.data[bucket].append(self._point_to_dict(record))
    
    def _point_to_dict(self, point):
        """Convert a Point to a dictionary."""
        return {
            "measurement": point._measurement,
            "tags": point._tags,
            "fields": point._fields,
            "time": point._time
        }


class MockQueryApi:
    """Mock implementation of InfluxDB QueryApi."""
    
    def __init__(self, write_api):
        """Initialize with reference to write_api."""
        self.write_api = write_api
    
    def query(self, query):
        """Execute a mock query."""
        # This is a very simplified implementation that just returns some data
        # based on the query string
        
        if "schema.tagValues" in query:
            # Handle tag values query
            if "tag: \"symbol\"" in query:
                # Return list of symbols
                return [MockTable([
                    MockRecord({"_value": "AAPL"}),
                    MockRecord({"_value": "MSFT"}),
                    MockRecord({"_value": "GOOGL"})
                ])]
            elif "tag: \"model_name\"" in query:
                # Return list of models
                return [MockTable([
                    MockRecord({"_value": "test_model"})
                ])]
        
        # Default empty result
        return []
    
    def query_data_frame(self, query):
        """Execute a mock query and return as DataFrame."""
        # Extract measurement from query
        measurement = None
        if "r._measurement == \"stock_data\"" in query:
            measurement = "stock_data"
        elif "r._measurement == \"model_params\"" in query:
            measurement = "model_params"
        elif "r._measurement == \"model_training_history\"" in query:
            measurement = "model_training_history"
        
        # Extract symbol from query
        symbol = None
        if "r.symbol ==" in query:
            # Extract symbol from query string
            start = query.find("r.symbol ==") + 12
            end = query.find("\"", start + 1)
            symbol = query[start:end]
        
        # Extract model_name from query
        model_name = None
        if "r.model_name ==" in query:
            # Extract model_name from query string
            start = query.find("r.model_name ==") + 15
            end = query.find("\"", start + 1)
            model_name = query[start:end]
        
        # Extract version from query
        version = None
        if "r.version ==" in query:
            # Extract version from query string
            start = query.find("r.version ==") + 12
            end = query.find("\"", start + 1)
            version = query[start:end]
        
        # Extract status from query
        status = None
        if "r.status ==" in query:
            # Extract status from query string
            start = query.find("r.status ==") + 12
            end = query.find("\"", start + 1)
            status = query[start:end]
        
        # Find matching records
        records = []
        for bucket, bucket_data in self.write_api.data.items():
            for record in bucket_data:
                if record["measurement"] != measurement:
                    continue
                
                if symbol and record["tags"].get("symbol") != symbol:
                    continue
                
                if model_name and record["tags"].get("model_name") != model_name:
                    continue
                
                if version and record["tags"].get("version") != version:
                    continue
                
                if status and record["tags"].get("status") != status:
                    continue
                
                records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        # Convert records to DataFrame
        data = []
        for record in records:
            row = {
                "_time": record["time"],
                **record["tags"],
                **record["fields"]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add required columns for stock_data
        if measurement == "stock_data" and not df.empty:
            if "open" in df.columns:
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })
                df["result"] = "result"
                df["table"] = 0
                df["_start"] = df["_time"] - pd.Timedelta(days=1)
                df["_stop"] = df["_time"] + pd.Timedelta(days=1)
                df["_measurement"] = "stock_data"
        
        return df


class MockTable:
    """Mock implementation of InfluxDB Table."""
    
    def __init__(self, records):
        """Initialize with records."""
        self.records = records


class MockRecord:
    """Mock implementation of InfluxDB Record."""
    
    def __init__(self, values):
        """Initialize with values."""
        self.values = values


class MockPoint:
    """Mock implementation of InfluxDB Point."""
    
    def __init__(self, measurement):
        """Initialize with measurement."""
        self._measurement = measurement
        self._tags = {}
        self._fields = {}
        self._time = datetime.utcnow()
    
    def tag(self, key, value):
        """Add a tag."""
        self._tags[key] = value
        return self
    
    def field(self, key, value):
        """Add a field."""
        self._fields[key] = value
        return self
    
    def time(self, time):
        """Set time."""
        self._time = time
        return self


class MockInfluxDBClient:
    """Mock implementation of InfluxDB Client."""
    
    def __init__(self, url, token, org):
        """Initialize with connection parameters."""
        # Accept any URL scheme for testing
        if url.startswith("mock://"):
            url = url.replace("mock://", "http://")
            
        self.url = url
        self.token = token
        self.org = org
        self._write_api = MockWriteApi()
        self._buckets_api = MockBucketsApi()
        self._query_api = MockQueryApi(self._write_api)
    
    def write_api(self, write_options=None):
        """Return write API."""
        return self._write_api
    
    def query_api(self):
        """Return query API."""
        return self._query_api
    
    def buckets_api(self):
        """Return buckets API."""
        return self._buckets_api


# Export Point class for use in tests
Point = MockPoint